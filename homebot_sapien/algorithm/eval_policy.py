import json
import numpy as np
import imageio
import os
import sys
from typing import Any
import torch
from collections import deque
from homebot_sapien.algorithm.imitation.bc_utils.make_utils import (
    make_policy_from_config,
    make_dataset_from_config,
)
from homebot_sapien.algorithm.imitation.dataset import OBS_NORMALIZE_PARAMS
from homebot_sapien.utils.make_env import make_vec_env
from homebot_sapien.utils.math import quat2euler


# class ObsConverter(object):
#     """
#     Convert environment observation to the neural network input according to config
#     Assume called step by step
#     """

#     def __init__(self, config, device):
#         self.config = config
#         self.device = device
#         self.wrist_image_history = deque(maxlen=config["data_config"]["n_images"])
#         self.head_image_history = deque(maxlen=config["data_config"]["n_images"])

#     def clear_history(self):
#         self.wrist_image_history.clear()
#         self.head_image_history.clear()

#     def convert_obs(self, obs: dict) -> Any:
#         wrist_rgb = np.transpose(
#             obs["wrist-rgb"][0], (2, 0, 1)
#         )  # get rid of vec dimension
#         head_rgb = np.transpose(obs["third-rgb"][0], (2, 0, 1))
#         if len(self.wrist_image_history) == 0:
#             for _ in range(self.wrist_image_history.maxlen):
#                 self.head_image_history.append(head_rgb)
#                 self.wrist_image_history.append(wrist_rgb)
#         else:
#             self.head_image_history.append(head_rgb)
#             self.wrist_image_history.append(wrist_rgb)
#         history_wrist_rgb = np.array(self.wrist_image_history)
#         history_head_rgb = np.array(self.head_image_history)
#         if self.config["data_config"]["image_wrist_or_head"] == "wrist":
#             mb_imgs = history_wrist_rgb
#         elif self.config["data_config"]["image_wrist_or_head"] == "head":
#             mb_imgs = history_head_rgb
#         elif self.config["data_config"]["image_wrist_or_head"] == "both":
#             mb_imgs = np.concatenate([history_wrist_rgb, history_head_rgb], axis=0)
#         else:
#             raise NotImplementedError
#         mb_imgs = (
#             torch.from_numpy(np.expand_dims(mb_imgs, axis=0)).float().to(self.device)
#         )

#         robot_states = np.zeros(0)
#         if len(self.config["data_config"]["robot_state_keys"]):
#             for key in self.config["data_config"]["robot_state_keys"]:
#                 if key == "pose":
#                     tcp_pose = np.concatenate(
#                         [obs["tcp_pose"][0][:3], quat2euler(obs["tcp_pose"][0][3:])]
#                     )
#                     robot_states = np.concatenate([robot_states, tcp_pose])
#                 elif key == "gripper_width":
#                     gripper_width = np.array([obs["gripper_width"][0]])
#                     robot_states = np.concatenate([robot_states, gripper_width])
#                 elif key == "joint":
#                     joint = obs["robot_joints"][0]
#                     robot_states = np.concatenate([robot_states, joint])
#             state_mean = np.concatenate(
#                 [
#                     OBS_NORMALIZE_PARAMS[key]["mean"]
#                     for key in self.config["data_config"]["robot_state_keys"]
#                 ]
#             )
#             state_scale = np.concatenate(
#                 [
#                     OBS_NORMALIZE_PARAMS[key]["scale"]
#                     for key in self.config["data_config"]["robot_state_keys"]
#                 ]
#             )
#             robot_states = (robot_states - state_mean) / state_scale
#         mb_robot_states = (
#             torch.from_numpy(np.expand_dims(robot_states, axis=0))
#             .float()
#             .to(self.device)
#         )

#         mb_lang = []
#         return mb_imgs, mb_lang, mb_robot_states

#     def convert_action(self, action: np.ndarray):
#         # convert neural network output to acceptable actions for the environment
#         assert len(action.shape) == 2
#         _start_idx = 0
#         base_action = np.zeros((1, 2))
#         gripper_action = np.zeros((1, 1))
#         pose_action = np.zeros((1, 6))
#         for key in self.config["data_config"]["action_keys"]:
#             if key == "gripper":
#                 gripper_action = action[:, _start_idx : _start_idx + 1]
#                 _start_idx += 1
#             elif key == "pose":
#                 pose_action = action[:, _start_idx : _start_idx + 6]
#                 _start_idx += 6
#             else:
#                 raise NotImplementedError
#         return np.concatenate([base_action, pose_action, gripper_action], axis=-1)


def main(checkpoint: str, save_video=False):
    # policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = os.path.join(os.path.dirname(checkpoint), "config.txt")
    with open(config_file, "r") as f:
        config = json.loads(f.read())
    policy = make_policy_from_config(config, device)
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"], strict=False)
    policy.eval()
    env = make_vec_env(
        "Opendoor-v0",
        num_workers=1,
        done_when_success=True,
        io_config=config,
        kwargs={"action_relative": config["data_config"]["action_relative"]},
    )

    # obs
    # obs_converter = ObsConverter(config, device)

    if save_video:
        video_filename = "bc_eval"
        video_writer = imageio.get_writer(
            f"{video_filename}.mp4",
            fps=40,
            format="FFMPEG",
            codec="h264",
        )
    obs, _ = env.reset()
    n_desired_traj = 10
    n_traj = 0
    success_buffer = []
    assert env.num_envs == 1
    while n_traj < n_desired_traj:
        if save_video:
            video_writer.append_data(
                np.transpose(obs["image"][0][0], (1, 2, 0)).astype(np.uint8)
            )
        mb_imgs = torch.from_numpy(obs["image"]).to(device)
        mb_lang = obs["lang"]
        mb_robot_states = torch.from_numpy(obs["robot_state"]).to(device)
        with torch.no_grad():
            action = (
                policy.inference(mb_imgs, mb_lang, mb_robot_states, deterministic=False)
                .cpu()
                .numpy()
            )
        sim_action = action
        obs, reward, done, truncated, info = env.step(sim_action)
        if done[0] or truncated[0]:
            success_buffer.append(info[0]["is_success"])
            n_traj += 1
    if save_video:
        video_writer.close()
    print("success_buffer", success_buffer)


if __name__ == "__main__":
    main(sys.argv[1], save_video=True)
