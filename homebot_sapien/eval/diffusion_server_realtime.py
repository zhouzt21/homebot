
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio

import sys
sys.path.append("/home/zhouzhiting/Projects")

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat, mat2quat
from typing import List
from homebot.homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from diffusion_policy.policy import DiffusionPolicy
import torchvision.transforms as transforms
import pickle

import requests

from flask import Flask, jsonify, request
app = Flask(__name__)


############### init policy ###############
device = "cuda" if torch.cuda.is_available() else "cpu"

OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/home/zhouzhiting/Data/panda_data/cano_policy_pd_2/", "norm_stats_1.pkl"), "rb"))

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

ckpt_path = "/home/zhouzhiting/Data/panda_data/diffusion_policy_checkpoints/20250307_191102/policy_step_250000_seed_0.ckpt"

policy = DiffusionPolicy(policy_config)
policy.deserialize(torch.load(ckpt_path))
policy.eval()
policy.cuda()

##############################################

def process_action(action):
    """
        unnormalize the action to the original scale, prepare for the env step.
        input: action: (10,), float, np
        output: action: (10,), float, np
    """
    action = action * np.expand_dims(pose_gripper_scale, axis=0) + np.expand_dims(pose_gripper_mean, axis=0)
    return action

def process_data(all_cam_images, proprio_state):
    """
        process the data for diffusion policy model. 
        input:  all_cam_images: (M, h, w, c),  uint8, np (M is the number of cameras)
                proprio_state: (10,), float, np
        output: image_data: (M, c, h, w),  normalized in [0,1], float, np
                qpos_data: (10)
    """
    # all_cam_images = np.stack(image_list, axis=0) # already in shape (B, h, w, c)
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

def get_action(ldm_data):
    """
        state + image --> action
        communication with ldm server, and post action result to robot server.
    """
    all_cam_images = ldm_data["samples"]  # (M, h, w, c), uint8, np

    pose = ldm_data["tcp_pose"]   
    if pose.shape[0] == 7:
        pose_rot = pose[3:7]
        pose = np.concatenate([pose[:3], quat2euler(pose_rot)])

    gripper = ldm_data["gripper_width"]
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
            np.array([gripper]),
        ]
    )

    image_data, qpos_data = process_data(all_cam_images, proprio_state)
    image_data, qpos_data = image_data.cuda().unsqueeze(0), qpos_data.cuda().unsqueeze(0)

    # model output actions
    pred_actions = policy(qpos_data, image_data).squeeze().cpu()
    actions = process_action(pred_actions)

    converted_actions = []

    # count 10 steps for action
    for i in range(10):
        action = actions[i]
        mat_6 = action[3:9].reshape(3, 2)
        mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
        mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
        z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
        mat = np.c_[mat_6, z_vec]
        # assert mat.shape == (3, 3)

        pos = action[:3]
        gripper_width = action[-1]

        init_to_desired_pose = pose_at_obs @ get_pose_from_rot_pos(
            mat, pos
        )
        # quaternion, 8-dim
        pose_action = np.concatenate(
            [
                init_to_desired_pose[:3, 3],
                mat2quat(init_to_desired_pose[:3, :3]),  
                [gripper_width]
            ]
        )
        converted_actions.append(pose_action)

    res_dict = {
        'actions': converted_actions
    }
        
    return res_dict
    

@app.route("/diffusion_real", methods=["GET", "POST"])
def handle_request():
    if request.method == "GET":
        response = {"message": "GET response"}
        return jsonify(response)
    elif request.method == "POST":
        try:
            # Send request to ldm server
            ldm_server_url = "http://0.0.0.0:9977/ldm_real"
            ldm_response = requests.post(ldm_server_url, json=request.json)
            ldm_response.raise_for_status()   # check?
            ldm_data = ldm_response.json()

            # use diffusion model to get action
            res_dict = get_action(ldm_data)

            return jsonify(res_dict)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9988")  # check?