import copy
import glob
import importlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shutil
import time
import torch
import torch.nn as nn
import wandb
import sys
from collections import deque
from .imitation.dataset import (
    preprocess_demonstration,
    step_collate_fn,
    play_demonstration,
    OBS_NORMALIZE_PARAMS,
)
from .imitation.bc_utils.make_utils import (
    make_policy_from_config,
    make_dataset_from_config,
)
from datetime import datetime
from torch.utils.data import DataLoader


def adjust_lr(optimizer, base_lr, cur_epoch, warmup_epoch, num_epoch):
    if cur_epoch < warmup_epoch:
        lr = base_lr * cur_epoch / warmup_epoch
    else:
        lr = (
            base_lr
            * 0.5
            * (
                1.0
                + np.cos(
                    np.pi * (cur_epoch - warmup_epoch) / (num_epoch - warmup_epoch)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def get_optimizer_groups(model, default_wd):
    param_group_names, param_group_vars = dict(), dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ks = [k for (k, x) in enumerate(["bn", "ln", "norm", "bias", ""]) if x in n]
        if "token" in n:
            name_apx = "t"
            wd_val = 0.0
        elif "pos_embed" in n:
            name_apx = "p"
            wd_val = 0.0
        elif "bn" in n or "ln" in n or "norm" in n:
            name_apx = "n"
            wd_val = 0.0
        elif "bias" in n:
            name_apx = "b"
            wd_val = 0.0
        else:
            name_apx = "w"
            wd_val = default_wd

        param_group = f"wd:{name_apx}"
        if param_group not in param_group_names:
            item = {"params": [], "weight_decay": wd_val}
            param_group_names[param_group] = copy.deepcopy(item)
            param_group_vars[param_group] = copy.deepcopy(item)
        param_group_names[param_group]["params"].append(n)
        param_group_vars[param_group]["params"].append(p)

    param_list = list(param_group_vars.values())

    param_group_str = json.dumps(param_group_names, sort_keys=True, indent=2)
    print("Parameter groups:\n" + param_group_str)

    return param_list


def prepare_dataset(config):
    if not config["data_config"]["overwrite"]:
        return
    if isinstance(config["data_config"]["file_pattern"], str):
        config["data_config"]["file_pattern"] = [config["data_config"]["file_pattern"]]
    all_file_names = []
    for pattern in config["data_config"]["file_pattern"]:
        all_file_names.extend(list(glob.glob(pattern)))
    train_ids = np.random.choice(
        len(all_file_names), size=int(len(all_file_names) * 0.9), replace=False
    )
    eval_ids = [i for i in range(len(all_file_names)) if not i in train_ids]
    train_files = [all_file_names[i] for i in train_ids]
    eval_files = [all_file_names[i] for i in eval_ids]
    print("N traj train", len(train_files), "N traj eval", len(eval_files))
    if os.path.exists("tmp_dataset"):
        ans = input("Going to remove tmp_dataset [Y|n]")
        if ans == "Y":
            shutil.rmtree("tmp_dataset")
        else:
            exit()
    for file_name in train_files:
        play_demonstration(
            file_name,
            out_folder="tmp_dataset/train",
            # relative_action=True,
            # gripper_action=config["data_config"]["gripper_action"],
            # pose_or_joint="pose",
        )
        # preprocess_demonstration(file_name, out_folder="tmp_dataset")
    for file_name in eval_files:
        play_demonstration(
            file_name,
            out_folder="tmp_dataset/eval",
            # relative_action=True,
            # gripper_action=config["data_config"]["gripper_action"],
            # pose_or_joint="pose",
        )
    return


def train(config):
    prepare_dataset(config)
    runname = config.get("run_name", "test")
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{config['log_dir']}/{stamp}_{runname}"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(json.dumps(config))
    wandb.init(project="homebot-sim-bc", name=runname, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = make_policy_from_config(config, device)
    # optimizer = torch.optim.Adam(policy.parameters(), lr=base_lr)
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(policy, config["optim"]["weight_decay"]),
        lr=config["optim"]["base_lr"],
        weight_decay=config["optim"]["weight_decay"],
    )
    for name, param in policy.named_parameters():
        print(name, param.shape, param.requires_grad)
    # traj = []
    bc_dataset = make_dataset_from_config(config, "tmp_dataset/train")
    bc_dataloader = DataLoader(
        bc_dataset,
        config["optim"]["batch_size"],
        shuffle=True,
        collate_fn=step_collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    print("N data", len(bc_dataset))
    eval_dataset = make_dataset_from_config(config, "tmp_dataset/eval")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=step_collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    best_eval_loss = np.inf
    best_eval_epoch = -1
    # max_action = -np.inf * np.ones(n_action)

    train_step_count = 0
    for epoch in range(config["optim"]["stop_epoch"]):
        policy.eval()
        losses = []
        grad_norms = []
        eval_losses = []
        eval_action_dist = []
        for mb in eval_dataloader:
            # mb["action"] = mb["action"][:, :n_action]
            mb_imgs = torch.from_numpy(mb["rgb"]).to(device)
            mb_lang = mb["lang"]
            mb_robot_states = torch.from_numpy(mb["robot_state"]).float().to(device)
            mb_expert_actions = torch.from_numpy(mb["action"]).float().to(device)
            with torch.no_grad():
                eval_loss = policy.compute_bc_loss(
                    mb_imgs, mb_lang, mb_robot_states, mb_expert_actions
                )
                pred_action = (
                    policy.inference(
                        mb_imgs, mb_lang, mb_robot_states, deterministic=False
                    )
                    .cpu()
                    .numpy()
                )
            eval_losses.append(eval_loss.item())
            eval_action_dist.append(np.abs(pred_action - mb["action"]).squeeze())
        eval_action_dist = np.array(eval_action_dist)
        print("mean eval loss", np.mean(eval_losses))
        print("eval: pred action", pred_action[0], "gt action", mb_expert_actions[0])
        wandb.log({"Eval loss": np.mean(eval_losses)}, step=train_step_count)
        for i in range(eval_action_dist.shape[1]):
            wandb.log(
                {"Eval dist %d" % i: np.mean(eval_action_dist, axis=0)[i]},
                step=train_step_count,
            )
        if np.mean(eval_losses) < best_eval_loss:
            torch.save(
                {"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(log_dir, "bc_model.pt"),
            )
            best_eval_loss = np.mean(eval_losses)
            best_eval_epoch = epoch
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(
                {"policy": policy.state_dict(), "optimizer": optimizer.state_dict()},
                os.path.join(log_dir, "bc_model_%d.pt" % epoch),
            )
        lr = adjust_lr(
            optimizer,
            config["optim"]["base_lr"],
            epoch,
            config["optim"]["warmup_epoch"],
            config["optim"]["n_epoch"],
        )
        # all_indices = np.arange(len(traj))
        # np.random.shuffle(all_indices)
        policy.train()
        for batch_idx, mb in enumerate(bc_dataloader):
            # mb_idx = all_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            # mb_imgs = torch.from_numpy(np.array([traj[i]["rgb"] for i in mb_idx])).to(
            #     device
            # )
            # mb["action"] = mb["action"][:, :n_action]
            # max_action = np.maximum(max_action, np.max(np.abs(mb["action"]), axis=0))
            mb_imgs = torch.from_numpy(mb["rgb"]).to(device)
            # mb_lang = [traj[i]["lang"] for i in mb_idx]
            mb_lang = mb["lang"]
            # mb_robot_states = (
            #     torch.from_numpy(np.array([traj[i]["robot_state"] for i in mb_idx]))
            #     .float()
            #     .to(device)
            # )
            mb_robot_states = torch.from_numpy(mb["robot_state"]).float().to(device)
            # mb_expert_actions = (
            #     torch.from_numpy(np.array([traj[i]["action"] for i in mb_idx]))
            #     .float()
            #     .to(device)
            # )
            mb_expert_actions = torch.from_numpy(mb["action"]).float().to(device)
            data_weights = torch.ones((mb_imgs.shape[0], 1), dtype=torch.float32).to(
                device
            )
            # data_weights[
            #     torch.where(torch.any(mb_expert_actions[:, :2] > 0, dim=-1))[0]
            # ] = 5.0
            loss = policy.compute_bc_loss(
                mb_imgs, mb_lang, mb_robot_states, mb_expert_actions
            )  # This is actually not true for rotation prediction
            optimizer.zero_grad()
            loss.backward()
            # TODO: gradient norm?
            if config["optim"]["max_grad_norm"] > 0:
                total_norm = nn.utils.clip_grad_norm_(
                    policy.parameters(), config["optim"]["max_grad_norm"]
                )
                grad_norms.append(total_norm.item())
            optimizer.step()
            train_step_count += 1
            losses.append(loss.item())
            # log
        if len(losses):
            print("Epoch", epoch, "loss", np.mean(losses))
            wandb.log({"lr": lr}, step=train_step_count)
            wandb.log(
                {"Train loss": np.mean(losses)},
                step=train_step_count,
            )
            wandb.log({"Epoch": epoch}, step=train_step_count)
            with torch.no_grad():
                pred_action = policy.inference(
                    mb_imgs[0:1],
                    mb_lang[0:1],
                    mb_robot_states[0:1],
                    deterministic=False,
                )
            print("gt action", mb_expert_actions[0], "pred action", pred_action[0])
    os.rename(
        os.path.join(log_dir, "bc_model.pt"),
        os.path.join(
            log_dir, f"bc_model_val_epoch{best_eval_epoch}_{best_eval_loss}.pt"
        ),
    )


def eval(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = os.path.join(os.path.dirname(checkpoint), "config.txt")
    with open(config_file, "r") as f:
        config = json.loads(f.read())
    # n_images = 1
    # policy = MVPPolicy(state_dim=7, action_dim=7, n_images=n_images)
    # policy.to(device)
    policy = make_policy_from_config(config, device)
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    eval_dataset = make_dataset_from_config(
        config, "tmp_dataset/eval", file_sorted=True
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=step_collate_fn)
    gt_action_seq = []
    pred_action_seq = []
    step_count = 0
    with torch.no_grad():
        for mb in eval_dataloader:
            step_count += 1
            if step_count >= 300:
                break
            test_img = torch.from_numpy(mb["rgb"]).to(device)
            test_lang = mb["lang"]
            test_robot_state = torch.from_numpy(mb["robot_state"]).float().to(device)
            pred_action = policy.inference(
                test_img, test_lang, test_robot_state, deterministic=False
            )
            pred_action = pred_action[0].cpu().numpy()
            gt_action = mb["action"][0]
            # print("gt", gt_action, "pred", pred_action)
            gt_action_seq.append(gt_action)
            pred_action_seq.append(pred_action)

            # print("lang", test_lang, "gt action", gt_action)
            # fig, ax = plt.subplots(1, mb["rgb"][0].shape[0])
            # for i in range(mb["rgb"][0].shape[0]):
            #     ax[i].imshow(np.transpose(mb["rgb"][0][i].astype(np.uint8), (1, 2, 0)))
            # plt.show()
            # plt.close(fig)
            # xyz_dist.append(pred_action[2: 5] - gt_action[2: 5])
            # rot_dist.append(pred_action[5: 8] - gt_action[5: 8])
            # gripper_dist.append(pred_action[1] - gt_action[1])
    # print("mean xyz dist", np.abs(np.array(xyz_dist)).mean(axis=0),
    #       "mean rot dist", np.abs(np.array(rot_dist)).mean(axis=0),
    #       "mean gripper dist", np.abs(np.array(gripper_dist)).mean())
    gt_action_seq = np.array(gt_action_seq)
    pred_action_seq = np.array(pred_action_seq)
    for i in range(len(gt_action_seq)):
        print(i, "gt", gt_action_seq[i], "pred", pred_action_seq[i])

    fig, ax = plt.subplots(3, 3)
    plt.suptitle(os.path.basename(os.path.dirname(checkpoint)).split("_")[0])
    for i in range(pred_action_seq.shape[-1]):
        r = i // 3
        c = i % 3
        ax[r][c].plot(gt_action_seq[:, i], label="gt")
        ax[r][c].plot(pred_action_seq[:, i], label="pred")
        ax[r][c].set_ylim(-1.0, 1.0)
    # ax[0][0].plot(gt_action_seq[:, 0], label="gt")
    # ax[0][0].plot(pred_action_seq[:, 0], label="pred")
    # ax[0][0].set_title("terminate")
    # ax[0][1].plot(gt_action_seq[:, 1], label="gt")
    # ax[0][1].plot(pred_action_seq[:, 1], label="pred")
    # ax[0][1].set_title("gripper")
    # for i in range(3):
    #     ax[1][i].plot(gt_action_seq[:, 2 + i], label="gt")
    #     ax[1][i].plot(pred_action_seq[:, 2 + i], label="pred")
    #     ax[2][i].plot(gt_action_seq[:, 5 + i], label="gt")
    #     ax[2][i].plot(pred_action_seq[:, 5 + i], label="pred")
    plt.savefig("figure.png")


if __name__ == "__main__":
    train_config = importlib.import_module(sys.argv[1]).train_config
    train(train_config)
    # eval("logs/2024-01-04-23-26-57_mvp_doornotexture_acttool_sameemb128_both_state_jointgripper/bc_model.pt")
