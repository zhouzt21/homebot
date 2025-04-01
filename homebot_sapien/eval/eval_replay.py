import os
import torch
import imageio
import numpy as np

import sys
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat

import pickle
from datetime import datetime

import torchvision.transforms as transforms

sys.path.append("/home/zhouzhiting/Projects")
from homebot.homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos
# from .drawer import PickAndPlaceEnv
from homebot.homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv

# in docker
# OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/root/data/cano_policy_1/", "norm_stats.pkl"), "rb"))
# OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/root/data/cano_drawer_0915/", "norm_stats_1.pkl"), "rb"))
OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/home/zhouzhiting/panda_data/cano_policy_pd_1/", "norm_stats_1.pkl"), "rb"))

pose_gripper_mean = np.concatenate(
    [
        OBS_NORMALIZE_PARAMS[key]["mean"]
        for key in ["pose", "gripper_width"]
    ]
)
pose_gripper_scale = np.concatenate(
    [
        OBS_NORMALIZE_PARAMS[key]["scale"]
        for key in ["pose", "gripper_width"]
    ]
)
proprio_gripper_mean = np.concatenate(
    [
        OBS_NORMALIZE_PARAMS[key]["mean"]
        for key in ["proprio_state", "gripper_width"]
    ]
)
proprio_gripper_scale = np.concatenate(
    [
        OBS_NORMALIZE_PARAMS[key]["scale"]
        for key in ["proprio_state", "gripper_width"]
    ]
)

# cameras = ['third', 'wrist']
cameras = ['third']
usages = ['obs']

policy_config = {
    'lr': 1e-5,
    'num_images': len(cameras) * len(usages),
    'action_dim': 10,
    'observation_horizon': 1,
    'action_horizon': 1,
    'prediction_horizon': 20,

    'global_obs_dim': 10,
    'num_inference_timesteps': 10,
    'ema_power': 0.75,
    'vq': False,
}
# ckpt_path = "/root/data/diffusion_policy_checkpoints/20240415_083018/policy_step_200000_seed_0.ckpt"  # two obs xarm
# ckpt_path = "/home/zhouzhiting/panda_data/diffusion_policy_checkpoints/20250228_211805/policy_step_2000_seed_0.ckpt"
# ckpt_path = "/home/zhouzhiting/panda_data/diffusion_policy_checkpoints/20250303_115710/policy_step_15000_seed_0.ckpt"

root_dir = "/home/zhouzhiting/panda_data/cano_policy_pd_2"

def process_action(action):
    action = action * np.expand_dims(pose_gripper_scale, axis=0) + np.expand_dims(pose_gripper_mean, axis=0)
    return action

def process_data(image_list, proprio_state):
    all_cam_images = np.stack(image_list, axis=0)
    image_data = torch.from_numpy(all_cam_images)
    image_data = torch.einsum('k h w c -> k c h w', image_data)

    try:
        k, c, h, w = image_data.shape
        transformations = [
            # transforms.CenterCrop((int(h * 0.95), int(w * 0.95))),
            transforms.RandomCrop((int(h * 0.95), int(w * 0.95))),
            # transforms.Resize((240, 320), antialias=True),
            transforms.Resize((224, 224), antialias=True),
        ]
        for transform in transformations:
            image_data = transform(image_data)
            # print(front.shape, goal.shape)
    except Exception as e:
        print(e)

    image_data = image_data / 255.0

    # qpos = np.array([0.], dtype=float)
    proprio_state = np.array(proprio_state)
    proprio_state = (
                            proprio_state - proprio_gripper_mean
                    ) / proprio_gripper_scale
    qpos_data = torch.from_numpy(proprio_state).float()

    return image_data, qpos_data



def eval_imitation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="tool", #"none",
        allow_dir=["fruit"]
    )

    cameras = ["third"] #, "wrist"
    # usage = ["obs"]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join("tmp", stamp)
    # save_dir = "try"
    # num_seeds = 10000
    num_eval = 2 #5
    os.makedirs(save_dir, exist_ok=True)

    from tqdm import tqdm
    

    for i_eval in range(num_eval):
        seed = i_eval + 5000
        # seed = 4
        env.reset(seed=seed)

        model_id_list = [None]

        for ep_id, model_id in enumerate(model_id_list):
            
            save_path = os.path.join(root_dir, f"seed_{seed}")
            ep_path = os.path.join(save_path, f"ep_{ep_id}")

            video_writer = {cam: imageio.get_writer(
                os.path.join(save_dir, f"seed_{seed}_ep_{ep_id}_cam_{cam}_replay.mp4"),
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            for step in tqdm(range(20)):
                obs = env.get_observation()
                image_list = []
                for cam in cameras:
                    image_list.append(obs[f"{cam}-rgb"])
                    # imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                    # video_writer[cam].append_data(obs[f"{cam}-rgb"])

                pkl_action = []
                for i in range(10):
                    pkl_path = os.path.join(ep_path, f"step_{step*10+i}.pkl")
                    pkl = pickle.load(open(pkl_path, "rb"))

                    if i == 0:
                        pkl_tcp_pose = pkl["tcp_pose"]
                        pkl_gripper_width = pkl["gripper_width"]
                    
                    pkl_action.append(pkl["action"] )
                    # print(pkl["action"])

                pose = pkl_tcp_pose #obs["tcp_pose"]
                pose_p, pose_q = pose[:3], pose[3:]
                pose_mat = quat2mat(pose_q)

                pose_at_obs = get_pose_from_rot_pos(
                    pose_mat, pose_p
                )

                pose_mat_6 = pose_mat[:, :2].reshape(-1)
                proprio_state = np.concatenate(
                    [
                        pose_p,
                        pose_mat_6,
                        np.array([pkl_gripper_width]), #obs["gripper_width"]
                    ]
                )

                image_data, qpos_data = process_data(image_list, proprio_state)
                image_data, qpos_data = image_data.cuda().unsqueeze(0), qpos_data.cuda().unsqueeze(0)
                # print(image_data.shape, qpos_data.shape)

                converted_actions = []

                for i in range(10):

                    action = pkl_action[i]
                    _, _, _, _, info = env.step(action)

                    obs = env.get_observation()
                    for cam in cameras:
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

            for writer in video_writer.values():
                writer.close()

            # exit()


if __name__ == "__main__":
    eval_imitation()
