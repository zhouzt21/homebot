import argparse
import imageio
import json
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from homebot_sapien.algorithm.bc import adjust_lr
from homebot_sapien.algorithm.imitation.dataset import EpisodeDataset, step_collate_fn
from homebot_sapien.algorithm.imitation.networks.image_state_policy import (
    DiffusionPolicy,
)
from homebot_sapien.algorithm.imitation.train_config import diffusion_config
from homebot_sapien.utils.make_env import make_vec_env
from tqdm import tqdm
from typing import Union


def train(config: dict):
    runname = config.get("run_name", "test")
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = f"{config['log_dir']}/{stamp}_{runname}"
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(json.dumps(config))
    wandb.init(project=config["project_name"], name=runname, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy(config["diffusion_config"])
    policy.to(device)
    if config.get("pretrained_checkpoint") is not None:
        checkpoint_dict = torch.load(config["pretrained_checkpoint"])
        policy.deserialize(checkpoint_dict["policy"])
    optimizer = torch.optim.AdamW(
        policy.nets.parameters(),
        lr=config["optim"]["base_lr"],
        weight_decay=config["optim"]["weight_decay"],
    )
    for name, param in policy.named_parameters():
        print(name, param.shape, param.requires_grad)
    # traj = []
    estimate_stats = (
        config["estimate_stats"] and config.get("pretrained_checkpoint") is None
    )
    bc_dataset = EpisodeDataset(
        [
            os.path.join(config["demo_root"][i], "train")
            for i in range(len(config["demo_root"]))
        ],
        chunk_size=config["diffusion_config"]["prediction_horizon"],
        data_config=config["data_config"],
        estimate_stats=estimate_stats,
    )
    if config.get("pretrained_checkpoint") is not None:
        checkpoint = config["pretrained_checkpoint"]
        if os.path.exists(
            os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl")
        ):
            with open(
                os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl"), "rb"
            ) as f:
                obs_normalize_params = pickle.load(f)
        else:
            obs_normalize_params = None
        bc_dataset.update_obs_normalize_params(obs_normalize_params)

    with open(os.path.join(log_dir, "normalize_params.pkl"), "wb") as f:
        pickle.dump(bc_dataset.OBS_NORMALIZE_PARAMS, f)

    bc_dataloader = DataLoader(
        bc_dataset,
        config["optim"]["batch_size"],
        shuffle=True,
        collate_fn=step_collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    print("N data", len(bc_dataset))
    eval_dataset = EpisodeDataset(
        [
            os.path.join(config["demo_root"][i], "eval")
            for i in range(len(config["demo_root"]))
        ],
        chunk_size=config["diffusion_config"]["prediction_horizon"],
        data_config=config["data_config"],
    )
    eval_dataset.update_obs_normalize_params(bc_dataset.OBS_NORMALIZE_PARAMS)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=32,
        collate_fn=step_collate_fn,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
    )
    transformations = None
    best_eval_loss = np.inf
    best_env_sr = -1
    best_eval_epoch = -1
    # max_action = -np.inf * np.ones(n_action)

    test_env = make_vec_env(
        "Opendoor-v0",
        num_workers=16,
        done_when_success=True,
        io_config=config,
        obs_normalize_params=bc_dataset.OBS_NORMALIZE_PARAMS,
        kwargs={
            "action_relative": config["data_config"]["action_relative"],
            "door_from_urdf": False,
            "need_door_shut": config["env_config"].get("need_door_shut", True),
            "use_real": config["env_config"].get("use_real", False),
            # "domain_randomize": False,
        },
    )

    train_step_count = 0
    for epoch in range(config["optim"]["stop_epoch"]):
        policy.eval()
        losses = []
        grad_norms = []
        eval_losses = []
        # eval_action_dist = []
        if epoch > 0:
            for mb in tqdm(eval_dataloader):
                # mb["action"] = mb["action"][:, :n_action]
                mb_imgs = torch.from_numpy(mb["rgb"]).to(device)
                mb_lang = mb["lang"]
                mb_robot_states = torch.from_numpy(mb["robot_state"]).float().to(device)
                mb_expert_actions = torch.from_numpy(mb["action"]).float().to(device)
                mb_is_pad = torch.from_numpy(mb["is_pad"]).to(device)
                with torch.no_grad():
                    eval_loss = policy.compute_bc_loss(
                        mb_imgs, mb_lang, mb_robot_states, mb_expert_actions, mb_is_pad
                    )
                    pred_action = (
                        policy.inference(
                            mb_imgs, mb_lang, mb_robot_states, deterministic=False
                        )
                        .cpu()
                        .numpy()
                    )
                eval_losses.append(eval_loss.item())
                # eval_action_dist.append(np.abs(pred_action - mb["action"]).squeeze())
            # eval_action_dist = np.array(eval_action_dist)
            print("mean eval loss", np.mean(eval_losses))
            print(
                "eval: pred action", pred_action[0], "gt action", mb_expert_actions[0]
            )
            wandb.log({"Eval loss": np.mean(eval_losses)}, step=train_step_count)
            if epoch % config.get("eval_interval", 1) == 0:
                env_success_rate = rollout(policy, test_env, n_desired_episode=50)
                wandb.log({"success rate": env_success_rate}, step=train_step_count)
                # for i in range(eval_action_dist.shape[1]):
                #     wandb.log(
                #         {"Eval dist %d" % i: np.mean(eval_action_dist, axis=0)[i]},
                #         step=train_step_count,
                #     )
                if env_success_rate > best_env_sr:
                    torch.save(
                        {"policy": policy.serialize()},
                        os.path.join(log_dir, "bc_model.pt"),
                    )
                    best_env_sr = env_success_rate
                    best_eval_epoch = epoch
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(
                {"policy": policy.serialize()},
                os.path.join(log_dir, "bc_model_%d.pt" % epoch),
            )
        # lr = config["optim"]["base_lr"]
        # all_indices = np.arange(len(traj))
        # np.random.shuffle(all_indices)
        policy.train()
        for batch_idx, mb in tqdm(enumerate(bc_dataloader)):
            lr = adjust_lr(
                optimizer,
                config["optim"]["base_lr"],
                epoch * len(bc_dataset) // config["optim"]["batch_size"] + batch_idx,
                config["optim"]["warmup_epoch"]
                * len(bc_dataset)
                // config["optim"]["batch_size"],
                config["optim"]["n_epoch"]
                * len(bc_dataset)
                // config["optim"]["batch_size"],
            )
            mb_imgs = torch.from_numpy(mb["rgb"]).to(device)
            if config["optim"]["augment_image"]:
                if transformations is None:
                    original_size = mb_imgs.shape[-2:]
                    ratio = 0.95
                    transformations = T.Compose(
                        [
                            T.RandomCrop(
                                size=[
                                    int(original_size[0] * ratio),
                                    int(original_size[1] * ratio),
                                ]
                            ),
                            T.Resize(original_size, antialias=True),
                            T.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                            T.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5
                            ),  # , hue=0.08)
                        ]
                    )
                bsz, n_image, img_size = (
                    mb_imgs.shape[0],
                    mb_imgs.shape[1],
                    mb_imgs.shape[2:],
                )
                mb_imgs = transformations(mb_imgs.view(bsz * n_image, *img_size)).view(
                    bsz, n_image, *img_size
                )
            mb_lang = mb["lang"]
            mb_robot_states = (
                torch.from_numpy(
                    mb["robot_state"]
                    + np.random.uniform(
                        -config["optim"]["robot_state_noise"],
                        config["optim"]["robot_state_noise"],
                        size=mb["robot_state"].shape,
                    )
                )
                .float()
                .to(device)
            )
            mb_expert_actions = torch.from_numpy(mb["action"]).float().to(device)
            mb_is_pad = torch.from_numpy(mb["is_pad"]).to(device)
            loss = policy.compute_bc_loss(
                mb_imgs, mb_lang, mb_robot_states, mb_expert_actions, mb_is_pad
            )
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
        os.path.join(log_dir, f"bc_model_sr_epoch{best_eval_epoch}_{best_env_sr}.pt"),
    )


def rollout(
    checkpoint: Union[str, DiffusionPolicy],
    env=None,
    n_desired_episode=10,
    save_video=False,
    inference_horizon=12,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(checkpoint, str):
        with open(os.path.join(os.path.dirname(checkpoint), "config.txt"), "r") as f:
            config = json.loads(f.read())
        # config["diffusion_config"]["num_inference_timesteps"] = 50 # same as training
        config["diffusion_config"]["inference_horizon"] = inference_horizon
        policy = DiffusionPolicy(config["diffusion_config"])
        policy.to(device)
        checkpoint_dict = torch.load(checkpoint)
        policy.deserialize(checkpoint_dict["policy"])
        policy.eval()
    else:
        policy = checkpoint
        policy.eval()
    if env is None:
        assert isinstance(checkpoint, str)
        if os.path.exists(
            os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl")
        ):
            with open(
                os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl"), "rb"
            ) as f:
                obs_normalize_params = pickle.load(f)
        else:
            obs_normalize_params = None
        env = make_vec_env(
            "Opendoor-v0",
            num_workers=16,
            done_when_success=True,
            io_config=config,
            obs_normalize_params=obs_normalize_params,
            return_middle_obs=save_video,
            kwargs={
                "action_relative": config["data_config"]["action_relative"],
                "door_from_urdf": False,
                "need_door_shut": config["env_config"].get("need_door_shut", True),
                "use_real": config["env_config"].get("use_real", False),
                # "domain_randomize": False,
            },
        )

    n_episode = 0
    success_buffer = []
    # inference_freq = config["diffusion_config"].get("inference_horizon", 10)
    if save_video:
        video_filename = "bc_eval"
        video_writer = imageio.get_writer(
            f"{video_filename}.mp4",
            fps=20,
            format="FFMPEG",
            codec="h264",
        )
    obs, info = env.reset()
    while n_episode < n_desired_episode:
        if save_video:
            if "obs_seq" in info[0]:
                for i in range(len(info[0]["obs_seq"])):
                    video_writer.append_data(
                        np.transpose(
                            info[0]["obs_seq"][i]["image"][-1], (1, 2, 0)
                        ).astype(np.uint8)
                    )
            else:
                video_writer.append_data(
                    np.transpose(obs["image"][0][-1], (1, 2, 0)).astype(np.uint8)
                )
        # if n_step % inference_freq == 0:
        mb_imgs = torch.from_numpy(obs["image"]).float().to(device)
        mb_lang = obs["lang"]
        mb_robot_states = torch.from_numpy(obs["robot_state"]).float().to(device)
        with torch.no_grad():
            pred_action_seq = (
                policy.inference(mb_imgs, mb_lang, mb_robot_states, deterministic=False)
                .cpu()
                .numpy()
            )
        # pred_action = pred_action_seq[:, n_step % inference_freq]
        pred_action = pred_action_seq.reshape((pred_action_seq.shape[0], -1))
        obs, reward, done, truncated, info = env.step(pred_action)
        for e_idx in range(env.num_envs):
            if done[e_idx] or truncated[e_idx]:
                n_episode += 1
                # print("is success:", info[e_idx]["is_success"])
                success_buffer.append(info[e_idx]["is_success"])
    if save_video:
        video_writer.close()
    print("mean success", np.mean(success_buffer))
    return np.mean(success_buffer)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--inference_horizon", type=int, default=12)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.play:
        train(diffusion_config.train_config)
    else:
        rollout(
            args.checkpoint,
            n_desired_episode=100,
            save_video=args.save_video,
            inference_horizon=args.inference_horizon,
        )
