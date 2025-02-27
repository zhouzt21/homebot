import gymnasium as gym
import torch
from .open_door import OpenDoorEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
gym.register(
    "Opendoor-v0",
    OpenDoorEnv,
    max_episode_steps=600,
    kwargs=dict(
        use_gui=False,
        device=device,
        obs_keys=(
            "tcp_pose",
            "gripper_width",
            "robot_joints",
            "wrist-rgb",
            "third-rgb",
            "privileged_obs",
        ),
        door_from_urdf=False,
    ),
)
