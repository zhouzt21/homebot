import json
import numpy as np
import imageio
import os
from typing import Any
import torch
from tqdm import tqdm 
import pickle

import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')
sys.path.append('/home/zhouzhiting/Projects')

from homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv

from diffusion_policy.datasets.dataset import load_sim2sim_data 
from diffusion_policy.utils.utils import compute_dict_mean, set_seed 
from diffusion_policy.policy import DiffusionPolicy  
from transforms3d.quaternions import  quat2mat
import torchvision.transforms as transforms
from transforms3d.euler import mat2euler


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

def get_pose_from_rot_pos(mat: np.ndarray, pos: np.ndarray):
    return np.concatenate(
        [
            np.concatenate([mat, pos.reshape(3, 1)], axis=-1),
            np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4),
        ],
        axis=0,
    )

def convert_action_to_7d(action_single):
    """Convert 10D action to 7D action (pos + euler + gripper)"""
    pos = action_single[:3]  # 位置 (3维)
    rot_mat_flat = action_single[3:9]  # 展平的旋转矩阵前两列 (6维)
    gripper = action_single[9:]  # 夹爪 (1维)
    
    # 重构旋转矩阵
    rot_mat = np.zeros((3, 3))
    rot_mat[:, 0] = rot_mat_flat[0:3]
    rot_mat[:, 1] = rot_mat_flat[3:6]
    rot_mat[:, 0] = rot_mat[:, 0] / np.linalg.norm(rot_mat[:, 0])
    rot_mat[:, 1] = rot_mat[:, 1] / np.linalg.norm(rot_mat[:, 1])
    # 通过叉乘计算第三列
    rot_mat[:, 2] = np.cross(rot_mat[:, 0], rot_mat[:, 1])
    
    # 转换为欧拉角 (rx, ry, rz)
    euler = mat2euler(rot_mat)
    
    action_7d = np.concatenate([
        pos,           
        euler,        
        gripper        
    ])
    
    return action_7d


def make_policy(policy_config):
    policy = DiffusionPolicy(policy_config)
    return policy

def forward_pass(data, policy):
    images = data["images"]
    qpos = data["proprio_state"]
    action = data["action"]
    is_pad = data["is_pad"]
    # image_data, qpos_data, action_data, is_pad = data
    if action is None:
        images, qpos= images.cuda().unsqueeze(0), qpos.cuda().unsqueeze(0)
    else:
        images, qpos, action, is_pad = images.cuda(), qpos.cuda(), action.cuda(), is_pad.cuda()
    return policy(qpos, images, action, is_pad)


def eval_bc_loss(data_config, num_steps, policy):
    _, val_dataloader, _, norm = load_sim2sim_data(**data_config)
    print("num_steps: ", num_steps)

    print('validating')

    with torch.inference_mode():
        policy.eval()
        validation_dicts = []
        for batch_idx, data in enumerate(val_dataloader):
            # print("data[qpos].shape: ", data["proprio_state"].shape) #torch.Size([128, 1, 3, 224, 224])
            # data["action"] = None
            forward_dict = forward_pass(data, policy)
                
            validation_dicts.append(forward_dict)
            if batch_idx > 50:
                break

        validation_summary = compute_dict_mean(validation_dicts)

        epoch_val_loss = validation_summary['loss']
        print(f'Val loss:   {epoch_val_loss:.5f}')


    for k in list(validation_summary.keys()):
        validation_summary[f'val_{k}'] = validation_summary.pop(k)
    summary_string = ''
    for k, v in validation_summary.items():
        summary_string += f'{k}: {v.item():.3f} '
    print(summary_string)


def eval_maniskill(n_desired_traj, policy, save_video=True):
    #### make env 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    policy.eval()
    env =  PickAndPlaceEnv(
        use_gui=False,
        device=device_str,
        obs_keys=( "tcp_pose", "gripper_width"), #"image",
        domain_randomize=True,
        canonical=True,
        action_relative="none",
        allow_dir=["fruit"],
    )
    cameras = ["third"]

    seed = 5

    if save_video:
        video_filename = "eval_maniskill"
        video_writer = {cam: imageio.get_writer(
            f"{video_filename}_env_{seed}_mani.mp4",
            fps=20,
            format="FFMPEG",
            codec="h264",
        ) for cam in cameras}
    
    _, _ = env.reset(seed=seed)

    n_traj = 0

    traj_step =300 # n times of 10 steps

    n_desired_traj =1
    while n_traj < n_desired_traj:
        print("trajectory: ", n_traj)
        for _ in tqdm(range(traj_step)):

            ########### sim2sim episode dataset process
            obs = env.get_observation()
            image_list = []
            for cam in cameras:
                image_list.append(obs[f"{cam}-rgb"])
            all_cam_images = np.stack(image_list, axis=0)
            image_data = torch.from_numpy(all_cam_images)
                ## my 
            # images =  torch.from_numpy(obs["image"]).to(device)
            # print("images: ", images.shape) #torch.Size([240, 320, 3])
            # images = images.unsqueeze(0)

            images = torch.einsum('k h w c -> k c h w', image_data)  # images
            original_size = images.shape[2:]
            ratio = 0.95
            transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize((224, 224), antialias=True),
            ]
            for transform in transformations:
                images = transform(images)
            images = images.float() / 255.0  #torch.Size([1, 3, 224, 224]) need to be unsqueezed

            proprio_state = np.zeros((10,), dtype=np.float32)
            tcp_pose = obs["tcp_pose"]
            pose_p, pose_q = tcp_pose[:3], tcp_pose[3:]
            pose_mat = quat2mat(pose_q)
            pose_mat_6 = pose_mat[:, :2].reshape(-1)
            proprio_state[:] = np.concatenate(
                [
                    pose_p,
                    pose_mat_6,
                    np.array([obs["gripper_width"]]),
                ]
            )        
            proprio_state = np.array(proprio_state)
            proprio_state = (
                                    proprio_state - proprio_gripper_mean
                            ) / proprio_gripper_scale
            qpos_data = torch.from_numpy(proprio_state).float()


            pose_at_obs = get_pose_from_rot_pos(
                    pose_mat, pose_p
                )

            data = {}
            data["images"] = images
            data["proprio_state"] = qpos_data
            data["action"] = None
            data["is_pad"] = None

            with torch.no_grad():
                naction = forward_pass(data, policy)

            actions = naction.squeeze(0).cpu()  # shape: (20, 10)
            actions = process_action(actions)

            converted_actions = []
            for i in range(10):
                action = actions[i]
                mat_6 = action[3:9].reshape(3, 2)
                mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
                mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
                z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
                mat = np.c_[mat_6, z_vec]
    
                pos = action[:3]
                gripper_width = action[-1]

                init_to_desired_pose = pose_at_obs @ get_pose_from_rot_pos(
                    mat, pos
                )
                pose_action = np.concatenate(
                    [
                        init_to_desired_pose[:3, 3],
                        mat2euler(init_to_desired_pose[:3, :3]),
                        [gripper_width]
                    ]
                )

                converted_actions.append(pose_action)
                _, _, _, _, _ = env.step(pose_action)

                obs = env.get_observation()

                if save_video:
                    for cam in cameras:
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

                # delta_action = convert_action_to_7d(action)
                # converted_actions.append(delta_action)

                # obs, _, _, _, _ = env.step(delta_action)
        
        print("trajectory done")
        n_traj += 1
        seed += 1
        obs, _ = env.reset(seed=seed)

    if save_video:
        for cam in cameras:
            video_writer[cam].close()



if __name__ == "__main__":

    mode = 1
    set_seed(0)
    # command line parameters
    seed = 0  
    # batch_size = 128 
    batch_size = 32 # change
    chunk_size = 20
    # num_steps = 5000
    num_steps = 10
    n_desired_traj = 10
    resume_ckpt_path = "/home/zhouzhiting/panda_data/diffusion_policy_checkpoints/20250228_211805/policy_step_2000_seed_0.ckpt" # TODO
    data_roots = ["/home/zhouzhiting/panda_data/cano_policy_pd_1"]
    stats_path = os.path.join(data_roots[0], f"norm_stats_{len(data_roots)}.pkl")

    num_seeds = 500
    camera_names = ["third"]
    usages = ["obs"]

    data_config = {
        "data_roots": data_roots,
        "num_seeds": num_seeds, 
        "camera_names": camera_names,
        "usages": usages,
        "chunk_size": chunk_size,
        "train_batch_size": batch_size,
        "val_batch_size": batch_size,
        "norm_stats_path": stats_path
    }

    policy_config = {
        'lr': 1e-5,
        'num_images': len(camera_names) * len(usages),
        'action_dim': 10,
        'observation_horizon': 1,
        'action_horizon': 1,
        'prediction_horizon': chunk_size,
        'global_obs_dim': 10,
        'num_inference_timesteps': 10,
        'ema_power': 0.75,
        'vq': False,
    }

    set_seed(seed)
    policy = make_policy(policy_config)

    if os.path.exists(resume_ckpt_path):
        loading_status = policy.deserialize(torch.load(resume_ckpt_path))
        print(f'Resume policy from: {resume_ckpt_path}, Status: {loading_status}')
    else:
        raise Exception(f'Checkpoint at {resume_ckpt_path} not found!')

    policy.cuda()

    if mode == 0:
        eval_bc_loss( data_config, num_steps, policy)
    elif mode == 1:
        eval_maniskill(n_desired_traj, policy, save_video=True)