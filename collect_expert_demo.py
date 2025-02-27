import argparse
import cv2
import homebot_sapien.utils.make_env as make_env
import json
import numpy as np
import os
import pickle
import sapien.core as sapien
import shutil
import sys
import time
import torch
from tqdm import tqdm
from typing import Sequence, Optional
from homebot_sapien.utils.math import (
    quat2euler,
    euler2mat,
    get_pose_from_rot_pos,
    mat2euler,
)
from homebot_sapien.algorithm.imitation.networks.image_state_policy import (
    DiffusionPolicy,
)


def dump_frame(
    vec_obs: dict,
    file_names: Sequence[str],
    desired_pose: Optional[np.ndarray],
    desired_joints: Optional[np.ndarray],
    desired_gripper_width: Optional[np.ndarray],
):
    assert vec_obs["tcp_pose"].shape[0] == len(file_names)
    for i in range(len(file_names)):
        # Relative to the initial base frame
        relative_pose = sapien.Pose(
            vec_obs["tcp_pose"][i][:3], vec_obs["tcp_pose"][i][3:]
        )
        robot_xyz = relative_pose.p
        robot_rpy = quat2euler(relative_pose.q)
        rgb_head = (vec_obs["third-rgb"][i]).astype(np.uint8)
        rgb_wrist = (vec_obs["wrist-rgb"][i]).astype(np.uint8)
        robot_joints = vec_obs["robot_joints"][i]
        gripper_width = vec_obs["gripper_width"][i]
        with open(file_names[i], "ab") as f:
            obj_to_save = {
                "rgb_head": rgb_head,
                "rgb_wrist": rgb_wrist,
                "robot_xyz": robot_xyz,
                "robot_rpy": robot_rpy,
                "robot_joints": robot_joints,
                "gripper_width": gripper_width,
            }
            if desired_pose is not None:
                obj_to_save["next_desired_pose"] = desired_pose[i]
            if desired_joints is not None:
                obj_to_save["next_desired_joints"] = desired_joints[i]
            if desired_gripper_width is not None:
                obj_to_save["next_desired_gripper_width"] = desired_gripper_width[i]
            pickle.dump(obj_to_save, f)


def speed_test():
    env = make_env.make_vec_env(
        "Opendoor-v0",
        4,
        done_when_success=True,
    )
    env.seed(42)
    env.reset()
    n_steps = 0
    t0 = time.time()
    for i in range(1000):
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, reward, done, truncated, info = env.step(action)
        n_steps += env.num_envs
    duration = time.time() - t0
    print("n_envs", env.num_envs, "FPS", n_steps / duration)
    # 32, 682.51
    # 16, 658.21
    # 8, 541.03
    # 4, 336.46 / 320.05 (ibl)
    # 2, 231.86
    # 1, 141.73 / 86.32 (ibl)


def collect_self_demo(checkpoint: str, demo_folder: str, num_traj: int):
    with open(os.path.join(os.path.dirname(checkpoint), "config.txt"), "r") as f:
        config = json.loads(f.read())
    if os.path.exists(
        os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl")
    ):
        with open(
            os.path.join(os.path.dirname(checkpoint), "normalize_params.pkl"), "rb"
        ) as f:
            obs_normalize_params = pickle.load(f)
    else:
        obs_normalize_params = None
    # config["diffusion_config"]["num_inference_timesteps"] = 50 # same as training
    config["diffusion_config"]["inference_horizon"] = 12
    env = make_env.make_vec_env(
        "Opendoor-v0",
        num_workers=16,
        done_when_success=True,
        io_config=config,
        obs_normalize_params=obs_normalize_params,
        return_middle_obs=True,
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

    obs, info = env.reset()

    # Prepare filenames
    if os.path.exists(demo_folder):
        ans = input(f"Going to remove existing data at {demo_folder}. [Y|n]")
        if ans == "Y":
            shutil.rmtree(demo_folder)
        else:
            return
    os.makedirs(demo_folder, exist_ok=True)
    episode_indices = [0 for _ in range(env.num_envs)]
    file_names = [
        os.path.join(demo_folder, "traj-%d-rank%d.pkl" % (episode_indices[i], i))
        for i in range(env.num_envs)
    ]
    pbar = tqdm(total=num_traj)
    while True:
        mb_imgs = torch.from_numpy(obs["image"]).float().to(device)
        mb_lang = obs["lang"]
        mb_robot_states = torch.from_numpy(obs["robot_state"]).float().to(device)
        with torch.no_grad():
            pred_action_seq = (
                (policy.rl_pred(mb_robot_states, mb_imgs, use_averaged_model=True))
                .cpu()
                .numpy()
            )
        pred_action = pred_action_seq.reshape((pred_action_seq.shape[0], -1))
        new_obs, reward, done, truncated, info = env.step(pred_action)
        reset_signal = np.logical_or(done, truncated)
        is_success_signal = np.array(
            [info[i].get("is_success", False) for i in range(len(info))]
        )
        reset_indices = np.where(reset_signal)[0]
        reset_success_signal = np.logical_and(reset_signal, is_success_signal)
        reset_fail_signal = np.logical_and(reset_signal, ~is_success_signal)
        pbar.update(np.sum(reset_success_signal))
        for j in range(len(info[0]["obs_seq"])):
            # convert expert action to the format of next_desired_pose
            if "desired_relative_pose_seq" in info[0]:
                desired_pose = np.array(
                    [
                        get_pose_from_rot_pos(
                            euler2mat(
                                quat2euler(info[i]["desired_relative_pose_seq"][j][3:])
                            ),
                            info[i]["desired_relative_pose_seq"][j][:3],
                        )
                        for i in range(len(info))
                    ]
                )
            else:
                print("Warning, no desired_relative_pose_seq from env")
                desired_pose = None
            if "desired_joints_seq" in info[0]:
                desired_joints = np.array(
                    [info[i]["desired_joints_seq"][j] for i in range(len(info))]
                )
            else:
                print("Warning, no desired_joints_seq from env")
                desired_joints = None
            if "desired_gripper_width_seq" in info[0]:
                desired_gripper_width = np.array(
                    [info[i]["desired_gripper_width_seq"][j] for i in range(len(info))]
                )
            else:
                print("Warning, no desired_gripper_width_seq from env")
                desired_gripper_width = None
            dump_frame(
                obs, file_names, desired_pose, desired_joints, desired_gripper_width
            )
            obs = {}
            for key in info[0]["obs_seq"][j].keys():
                obs[key] = np.array(
                    [info[i]["obs_seq"][j][key] for i in range(len(info))]
                )

        obs = new_obs
        # Remove failed trajectories
        for i in np.where(reset_fail_signal)[0]:
            os.remove(file_names[i])
        # for key in obs:
        #     print(key, obs[key].shape)
        if np.any(done) or np.any(truncated):
            if np.sum(episode_indices) + len(reset_indices) >= num_traj:
                incomplete_indices = np.where(~reset_signal)[0]
                for i in incomplete_indices:
                    os.remove(file_names[i])
                break
            for i in np.where(reset_success_signal)[0]:
                episode_indices[i] += 1
                file_names[i] = os.path.join(
                    demo_folder, "traj-%d-rank%d.pkl" % (episode_indices[i], i)
                )

    # train eval split
    all_names = os.listdir(demo_folder)
    os.makedirs(os.path.join(demo_folder, "train"))
    os.makedirs(os.path.join(demo_folder, "eval"))
    for fname in all_names[: int(0.95 * len(all_names))]:
        os.rename(
            os.path.join(demo_folder, fname), os.path.join(demo_folder, "train", fname)
        )
    for fname in all_names[int(0.95 * len(all_names)) :]:
        os.rename(
            os.path.join(demo_folder, fname), os.path.join(demo_folder, "eval", fname)
        )


def main(demo_folder: str, num_traj: int):
    env = make_env.make_vec_env(
        "Opendoor-v0",
        16,
        done_when_success=True,
        kwargs={
            "use_real": True,
        },
    )
    env.seed(42)
    obs, info = env.reset()
    # Prepare filenames
    if os.path.exists(demo_folder):
        ans = input(f"Going to remove existing data at {demo_folder}. [Y|n]")
        if ans == "Y":
            shutil.rmtree(demo_folder)
        else:
            return
    os.makedirs(demo_folder, exist_ok=True)
    episode_indices = [0 for _ in range(env.num_envs)]
    file_names = [
        os.path.join(demo_folder, "traj-%d-rank%d.pkl" % (episode_indices[i], i))
        for i in range(env.num_envs)
    ]
    pbar = tqdm(total=num_traj)
    while True:
        # action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        action = np.array(env.env_method("expert_action", noise_scale=0.5))
        new_obs, reward, done, truncated, info = env.step(action)
        # init_agv_pose = [info[i]["init_agv_pose"] for i in range(len(info))]
        reset_signal = np.logical_or(done, truncated)
        is_success_signal = np.array(
            [info[i].get("is_success", False) for i in range(len(info))]
        )
        reset_indices = np.where(reset_signal)[0]
        reset_success_signal = np.logical_and(reset_signal, is_success_signal)
        reset_fail_signal = np.logical_and(reset_signal, ~is_success_signal)
        pbar.update(np.sum(reset_success_signal))
        # for i in reset_indices:
        #     for key in new_obs:
        #         new_obs[key][i] = info[i]["terminal_observation"][key]
        # convert expert action to the format of next_desired_pose
        if "desired_relative_pose" in info[0]:
            desired_pose = np.array(
                [
                    get_pose_from_rot_pos(
                        euler2mat(quat2euler(info[i]["desired_relative_pose"][3:])),
                        info[i]["desired_relative_pose"][:3],
                    )
                    for i in range(len(info))
                ]
            )
        else:
            print("Warning, no desired_relative_pose from env")
            desired_pose = None
        if "desired_joints" in info[0]:
            desired_joints = np.array(
                [info[i]["desired_joints"] for i in range(len(info))]
            )
        else:
            print("Warning, no desired_joints from env")
            desired_joints = None
        if "desired_gripper_width" in info[0]:
            desired_gripper_width = np.array(
                [info[i]["desired_gripper_width"] for i in range(len(info))]
            )
        else:
            print("Warning, no desired_gripper_width from env")
            desired_gripper_width = None
        dump_frame(obs, file_names, desired_pose, desired_joints, desired_gripper_width)
        obs = new_obs
        # Remove failed trajectories
        for i in np.where(reset_fail_signal)[0]:
            os.remove(file_names[i])
        # for key in obs:
        #     print(key, obs[key].shape)
        if np.any(done) or np.any(truncated):
            if np.sum(episode_indices) + len(reset_indices) >= num_traj:
                incomplete_indices = np.where(~reset_signal)[0]
                for i in incomplete_indices:
                    os.remove(file_names[i])
                break
            for i in np.where(reset_success_signal)[0]:
                episode_indices[i] += 1
                file_names[i] = os.path.join(
                    demo_folder, "traj-%d-rank%d.pkl" % (episode_indices[i], i)
                )
    # print("obs", obs, "reward", reward, "done", done, "truncated", truncated, "info", info)

    # train eval split
    all_names = os.listdir(demo_folder)
    os.makedirs(os.path.join(demo_folder, "train"))
    os.makedirs(os.path.join(demo_folder, "eval"))
    for fname in all_names[: int(0.95 * len(all_names))]:
        os.rename(
            os.path.join(demo_folder, fname), os.path.join(demo_folder, "train", fname)
        )
    for fname in all_names[int(0.95 * len(all_names)) :]:
        os.rename(
            os.path.join(demo_folder, fname), os.path.join(demo_folder, "eval", fname)
        )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--type", choices=["scripted", "learned"], default="scripted")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--demo_folder", type=str)
    parser.add_argument("--num_traj", type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.type == "scripted":
        main(args.demo_folder, args.num_traj)
    elif args.type == "learned":
        collect_self_demo(
            args.checkpoint, demo_folder=args.demo_folder, num_traj=args.num_traj
        )
