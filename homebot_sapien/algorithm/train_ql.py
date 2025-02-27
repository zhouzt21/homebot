# first fill the replay buffer with successful rollouts
# then warm up q network
# then start online learning
from copy import deepcopy
import matplotlib.pyplot as plt
import imageio
import json
import numpy as np
import os
import torch
from homebot_sapien.utils.make_env import make_vec_env
from homebot_sapien.algorithm.imitation.networks.image_state_policy import (
    DiffusionPolicy,
)
from homebot_sapien.algorithm.bc import adjust_lr
from homebot_sapien.algorithm.imitation.dataset import EpisodeDataset
from homebot_sapien.algorithm.qlearning.diffusion_ql import Diffusion_QL
from homebot_sapien.algorithm.qlearning.replay_buffer import (
    DictObsReplayBuffer,
    DictObsPrioritizedReplayBuffer,
)
from collections import deque
from tqdm import tqdm
from homebot_sapien.utils import logger
from typing import Optional


def fill_replay_buffer(
    env,
    policy: DiffusionPolicy,
    replay_buffer: DictObsReplayBuffer,
    device,
    desired_steps: int,
):
    is_success_buffer = deque(maxlen=100)
    episode_reward_buffer = deque(maxlen=100)
    obs, _ = env.reset()
    policy.eval()
    for i in tqdm(range(desired_steps // env.num_envs)):
        mb_imgs = torch.from_numpy(obs["image"]).float().to(device)
        mb_lang = obs["lang"]
        mb_robot_states = torch.from_numpy(obs["robot_state"]).float().to(device)
        with torch.no_grad():
            pred_action_seq = (
                policy.inference(mb_imgs, mb_lang, mb_robot_states, deterministic=False)
                .cpu()
                .numpy()
            )
        pred_action = pred_action_seq.reshape((pred_action_seq.shape[0], -1))
        new_obs, reward, done, truncated, info = env.step(pred_action)
        # obs_tp1 = new_obs.copy()  # ! critical bug
        obs_tp1 = deepcopy(new_obs)
        for e_idx, _info in enumerate(info):
            if done[e_idx] or truncated[e_idx]:
                is_success_buffer.append(_info["is_success"])
                episode_reward_buffer.append(_info["episode"]["r"])
                for key in obs:
                    obs_tp1[key][e_idx] = _info["terminal_observation"][key]
                assert (
                    np.linalg.norm(obs_tp1["image"][e_idx] - new_obs["image"][e_idx])
                    > 0.1
                )
        replay_buffer.extend(
            obs, pred_action_seq, reward, obs_tp1, np.logical_or(done, truncated)
        )
        obs = new_obs
    print(
        "Original rollout success rate",
        np.mean(is_success_buffer),
        "episode reward",
        np.mean(episode_reward_buffer),
    )
    # for i in range(40):
    #     obs_dict, action, reward, next_obs_dict, done = replay_buffer.storage[i]
    #     plt.imsave("obs_%d.png" % i, np.transpose(obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8))
    #     plt.imsave("next_obs_%d.png" % i, np.transpose(next_obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8))
    #     print("step", i, "action", action, "reward", reward, "done", done)


class RolloutManager(object):
    def __init__(self, env, debug_video=False):
        self.env = env
        self.traj_cache = [[] for _ in range(env.num_envs)]
        self.transition = [
            {
                "obs": None,
                "action": None,
                "reward": None,
                "next_obs": None,
                "done": None,
                "is_pad": None,
            }
            for _ in range(env.num_envs)
        ]
        self.is_success_buffer = deque(maxlen=100)
        self.episode_reward_buffer = deque(maxlen=100)
        self._obs, _ = env.reset()
        self.debug_video = debug_video
        self.debug_traj_idx = 0
        self.debug_video_writer = imageio.get_writer(
            f"debug_video_{self.debug_traj_idx}.mp4",
            fps=20,
            format="FFMPEG",
            codec="h264",
        )

    def step(
        self,
        policy,
        success_buffer: Optional[DictObsReplayBuffer],
        rollout_buffer: Optional[DictObsReplayBuffer],
        device,
        use_averaged_model=False,
    ):
        obs = self._obs
        for i in range(self.env.num_envs):
            self.transition[i]["obs"] = {key: obs[key][i] for key in obs}
        mb_imgs = torch.from_numpy(obs["image"]).float().to(device)
        mb_lang = obs["lang"]
        mb_robot_states = torch.from_numpy(obs["robot_state"]).float().to(device)
        with torch.no_grad():
            pred_action_seq = (
                (
                    policy.rl_pred(
                        mb_robot_states, mb_imgs, use_averaged_model=use_averaged_model
                    )
                )
                .cpu()
                .numpy()
            )
        pred_action = pred_action_seq.reshape((pred_action_seq.shape[0], -1))
        new_obs, reward, done, truncated, info = self.env.step(pred_action)
        obs_tp1 = deepcopy(new_obs)
        for e_idx, _info in enumerate(info):
            if done[e_idx] or truncated[e_idx]:
                for key in obs:
                    obs_tp1[key][e_idx] = _info["terminal_observation"][key]
                assert (
                    np.linalg.norm(obs_tp1["image"][e_idx] - new_obs["image"][e_idx])
                    > 0.1
                )
            self.transition[e_idx]["action"] = pred_action_seq[e_idx]
            self.transition[e_idx]["reward"] = reward[e_idx]
            self.transition[e_idx]["next_obs"] = {
                key: obs_tp1[key][e_idx] for key in obs_tp1
            }
            self.transition[e_idx]["done"] = done[e_idx] or truncated[e_idx]
            self.transition[e_idx]["is_pad"] = _info["is_pad"]
            self.traj_cache[e_idx].append(deepcopy(self.transition[e_idx]))
            if done[e_idx] or truncated[e_idx]:
                self.is_success_buffer.append(_info["is_success"])
                self.episode_reward_buffer.append(_info["episode"]["r"])
                if _info["is_success"]:
                    if success_buffer is not None:
                        for _transition in self.traj_cache[e_idx]:
                            success_buffer.add(
                                _transition["obs"],
                                _transition["action"],
                                _transition["reward"],
                                _transition["next_obs"],
                                _transition["done"],
                                _transition["is_pad"],
                            )
                            if self.debug_video and e_idx == 0:
                                self.debug_video_writer.append_data(
                                    np.transpose(
                                        _transition["obs"]["image"][-1], (1, 2, 0)
                                    ).astype(np.uint8)
                                )
                        if self.debug_video and e_idx == 0:
                            print(
                                "traj_idx",
                                self.debug_traj_idx,
                                "reward",
                                [
                                    _transition["reward"]
                                    for _transition in self.traj_cache[e_idx]
                                ],
                            )
                            self.debug_video_writer.close()
                            self.debug_traj_idx += 1
                            self.debug_video_writer = imageio.get_writer(
                                f"debug_video_{self.debug_traj_idx}.mp4",
                                fps=20,
                                format="FFMPEG",
                                codec="h264",
                            )
                if rollout_buffer is not None:
                    for _transition in self.traj_cache[e_idx]:
                        rollout_buffer.add(
                            _transition["obs"],
                            _transition["action"],
                            _transition["reward"],
                            _transition["next_obs"],
                            _transition["done"],
                            _transition["is_pad"],
                        )
                self.traj_cache[e_idx] = []
        self._obs = new_obs

    def debug(self, buffer: DictObsReplayBuffer, policy, suffix=""):
        for i in range(4):
            obs_dict, action, reward, next_obs_dict, done, is_pad = buffer.storage[i]
            # plt.imsave(
            #     "obs_%d.png" % i,
            #     np.transpose(obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8),
            # )
            # plt.imsave(
            #     "next_obs_%d.png" % i,
            #     np.transpose(next_obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8),
            # )
            print(
                "step",
                i,
                "robot_state",
                obs_dict["robot_state"],
                "image",
                obs_dict["image"].mean(),
                obs_dict["image"].std(),
                "action",
                action,
                "is_pad",
                is_pad,
            )
        # Still do not understand the difference between the expert trace and the self success trace
        for i in range(4):
            fig, ax = plt.subplots(3, 4)
            obs_dict, action, *_ = buffer.storage[i]
            plt.imsave(
                "obs_%d_%s.png" % (i, suffix),
                np.transpose(obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8),
            )
            for j in range(action.shape[-1]):
                r = j // 4
                c = j % 4
                ax[r][c].set_ylim(-1.0, 1.0)
                ax[r][c].plot(action[:, r * 4 + c], color="blue")
            device = torch.device("cuda")
            mb_imgs = (
                torch.from_numpy(obs_dict["image"]).unsqueeze(dim=0).float().to(device)
            )
            mb_robot_states = (
                torch.from_numpy(obs_dict["robot_state"])
                .unsqueeze(dim=0)
                .float()
                .to(device)
            )
            with torch.no_grad():
                pred_action_seq = (
                    (policy.rl_pred(mb_robot_states, mb_imgs, use_averaged_model=True))[
                        0
                    ]
                    .cpu()
                    .numpy()
                )
            for j in range(pred_action_seq.shape[-1]):
                r = j // 4
                c = j % 4
                ax[r][c].plot(pred_action_seq[:, 4 * r + c], color="red")
            obs_dict, action, *_ = buffer.storage[buffer._noflush_size + i]
            # plt.imsave("obs_self_%d.png" % i, np.transpose(obs_dict["image"][-1], (1, 2, 0)).astype(np.uint8))
            # for j in range(action.shape[-1]):
            #     r = j // 4
            #     c = j % 4
            #     ax[r][c].plot(action[:, r * 4 + c], color="red")
            plt.savefig("debug%d_%s.png" % (i, suffix))
            plt.close(fig)

    # def correct_action_rotation(self, action: np.ndarray):
    #     normed_rot6d = np.array([rot6d2mat(action[i][3: 9])[:, :2].reshape(-1) for i in range(action.shape[0])])
    #     correct_action = action.copy()
    #     correct_action[:, 3: 9] = normed_rot6d
    #     return correct_action


def fill_expert_buffer_from_dataset(
    expert_replay: DictObsReplayBuffer, dataset: EpisodeDataset
):
    for i in range(len(dataset)):
        data = dataset[i]
        obs = {
            "image": data["rgb"].astype(np.uint8),
            "robot_state": data["robot_state"].astype(np.float32),
        }
        action = data["action"].astype(np.float32)
        reward = 0
        next_obs = {
            "image": np.zeros_like(obs["image"]),
            "robot_state": np.zeros_like(obs["robot_state"]),
        }
        done = False
        is_pad = data["is_pad"].astype(bool)
        expert_replay.add(obs, action, reward, next_obs, done, is_pad)
        if expert_replay.is_full():
            break


def main():
    checkpoint = "logs/2024-01-24-15-26-52_friction5_100traj_noiselevel0.5_deepgrasp_usedesired_chunk20/bc_model_349.pt"
    log_dir = "logs/ql/il_expert0k_self6k"
    warm_up_critic_epochs = 500 * 0
    actor_bconly_epochs = 500_000
    freeze_policy_except_final_conv = False
    discount = 0.99
    tau = 0.005
    n_online_epochs = 500_000 * 0
    n_env_steps_per_epoch = 1
    n_update_per_epoch = 1
    with open(os.path.join(os.path.dirname(checkpoint), "config.txt"), "r") as f:
        config = json.loads(f.read())
    # config["diffusion_config"]["num_inference_timesteps"] = 50 # same as training
    config["diffusion_config"]["inference_horizon"] = 12
    env = make_vec_env(
        "Opendoor-v0",
        num_workers=16,
        done_when_success=True,
        io_config=config,
        kwargs={
            "action_relative": config["data_config"]["action_relative"],
            "door_from_urdf": False,
            # "domain_randomize": False,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionPolicy(config["diffusion_config"])
    policy.to(device)
    checkpoint_dict = torch.load(checkpoint)
    policy.deserialize(checkpoint_dict["policy"])

    if freeze_policy_except_final_conv:
        # for key in ["backbones", "pools", "linears"]:
        #     for param in policy.nets["policy"][key].parameters():
        #         param.requires_grad_(False)
        for name, param in policy.nets.named_parameters():
            if not "noise_pred_net.final_conv" in name:
                param.requires_grad_(False)

    # replay_buffer = DictObsReplayBuffer(size=4_000)
    # replay_buffer = DictObsPrioritizedReplayBuffer(size=1_000, alpha=0.6)
    self_success_buffer = DictObsReplayBuffer(size=6_000)
    success_buffer = DictObsReplayBuffer(size=0)
    rollout_manager = RolloutManager(env, debug_video=False)
    ql_trainer = Diffusion_QL(
        config["diffusion_config"],
        discount,
        tau,
        device,
        pretrained_actor=policy,
        # pretrained_actor=None,
        lr=1e-4,
        grad_norm=0,
        max_q_backup=False,
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger.configure(log_dir)
    os.system(
        f"cp {os.path.join(os.path.dirname(checkpoint), 'config.txt')} {os.path.join(log_dir, 'config.txt')}"
    )

    # fill_replay_buffer(env, policy, replay_buffer, device, 1000)
    expert_dataset = EpisodeDataset(
        os.path.join(config["demo_root"], "train"),
        config["diffusion_config"]["prediction_horizon"],
        config["data_config"],
    )
    if success_buffer.buffer_size > 0:
        fill_expert_buffer_from_dataset(success_buffer, expert_dataset)
    for _ in tqdm(range(20 * self_success_buffer.buffer_size // env.num_envs)):
        rollout_manager.step(policy, self_success_buffer, None, device, True)
        if self_success_buffer.is_full():
            break

    print("Filled self success buffer")
    # for i in range(warm_up_critic_epochs):
    #     metric = ql_trainer.train(
    #         replay_buffer, n_q_iterations=20, q_only=True, batch_size=48
    #     )
    #     print("offline %d metric" % i, metric)

    # self imitation iteractions
    for e_idx in range(300):
        lr = adjust_lr(ql_trainer.actor_optimizer, 1e-4, e_idx + 1, 10, 1000)
        batch_size = 32
        metric = ql_trainer.train_bc(
            success_buffer,
            self_success_buffer,
            batch_size,
            self_ratio=len(self_success_buffer)
            / (len(success_buffer) + len(self_success_buffer)),
        )
        logger.logkv("epoch", e_idx)
        logger.logkv("lr", lr)
        for key in metric:
            logger.logkv(key, metric[key])
        # evaluate success rate
        evaluate_rollout = RolloutManager(env)
        while len(evaluate_rollout.is_success_buffer) < 50:
            evaluate_rollout.step(ql_trainer.actor, None, None, device, True)
        eval_success_rate = np.mean(evaluate_rollout.is_success_buffer)
        logger.logkv("success rate", eval_success_rate)
        logger.dump_tabular()
        if (e_idx + 1) % 100 == 0:
            ql_trainer.save_model(os.path.join(log_dir, "model_%d.pt" % e_idx))

    # is_success_buffer = deque(maxlen=100)
    # episode_reward_buffer = deque(maxlen=100)
    # obs, _ = env.reset()
    for i in range(n_online_epochs):
        for step_idx in range(n_env_steps_per_epoch):
            rollout_manager.step(policy, success_buffer, replay_buffer, device, True)
        if i % 100 == 0:
            # print(
            #     "In online epoch %d" % i, "rollout success rate", np.mean(is_success_buffer), "episode reward", np.mean(episode_reward_buffer)
            # )
            logger.logkv("epoch", i)
            logger.logkv("rollout_success", np.mean(rollout_manager.is_success_buffer))
            logger.logkv(
                "rollout_episode_reward", np.mean(rollout_manager.episode_reward_buffer)
            )
        if i < actor_bconly_epochs:
            ql_trainer.eta = 0.0
        else:
            ql_trainer.eta = 1.0
        for j in range(n_update_per_epoch):
            metric = ql_trainer.train(
                replay_buffer,
                n_q_iterations=1,
                batch_size=32,
                demo_buffer=success_buffer,
            )
        if i % 100 == 0:
            # print("In online epoch %d" % i, "metric", metric)
            for key in metric:
                logger.logkv(key, metric[key])
            logger.dump_tabular()
        if i % 5000 == 0:
            ql_trainer.save_model(os.path.join(log_dir, "model_%d.pt" % i))


if __name__ == "__main__":
    main()
