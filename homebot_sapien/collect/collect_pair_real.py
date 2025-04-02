

import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import pickle 

import sys 
sys.path.append("/home/zhouzhiting/Projects/homebot")
# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

import json
import requests
from datetime import datetime

from homebot_sapien.env.pick_and_place_panda_real import PickAndPlaceEnv
# from Projects.homebot.config import PANDA_DATA
PANDA_DATA = "/home/zhouzhiting/Data/panda_data"


def collect_rand_and_cano_data():
    """
    Collect imitation data for the policy, from pick_and_place_panda env.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    allow_dir = ["along", "column"]  # v0

    rand_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=False,
        allow_dir=allow_dir
    )

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        allow_dir=allow_dir
    )

    envs = {
        "rand": rand_pick_env,
        "cano": cano_pick_env,
    }
    cameras = ["third"]

    save_dir =os.path.join(PANDA_DATA, "sim2sim_pd_3")

    num_seeds = 10000
    steps_per_obj = 400

    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    cnt_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        for idx, (env_key, env) in enumerate(envs.items()):
            env.reset(seed=seed)

            random_state = np.random.RandomState(seed=seed)

            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"seed_{seed}_env_{env_key}_cam_{cam}.mp4",
                    # fps=40,
                    fps=10,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            model_id_list = list(env.objs.keys())
            random_state.shuffle(model_id_list)

            frame_id = 0

            for model_id in model_id_list:

                try:

                    goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
                    goal_q_rand = random_state.uniform(-0.5, 0.5)

                    prev_privileged_obs = None
                    for step in range(steps_per_obj):

                        action, done, _ = env.expert_action(
                            noise_scale=0.2, obj_id=model_id,
                            goal_obj_pose=sapien.Pose(
                                p=np.concatenate([np.array([0.4, -0.2]) + goal_p_rand, [0.76]]),
                                q=euler2quat(np.array([0, 0, goal_q_rand]))
                            )
                        )
                        _, _, _, _, info = env.step(action)

                        # rgb_images = env.render_all()
                        if frame_id % 10 == 0:
                            world_tcp_pose = env._get_tcp_pose()
                            gripper_width = env._get_gripper_width()
                            privileged_obs = np.concatenate(
                                [
                                    world_tcp_pose.p,
                                    world_tcp_pose.q,
                                    [gripper_width],
                                ]
                            )
                            if prev_privileged_obs is not None and np.all(
                                    np.abs(privileged_obs - prev_privileged_obs) < 1e-4):
                                env.expert_phase = 0
                                break
                            prev_privileged_obs = privileged_obs

                            rgb_images = env.capture_images_new(cameras=cameras)

                            if seed < num_vid:
                                for cam in cameras:
                                    video_writer[cam].append_data(rgb_images[f"{cam}-rgb"])
  
                            save_path = os.path.join(save_dir, f"seed_{seed}")
                            os.makedirs(save_path, exist_ok=True)
                            for cam in cameras:
                                imageio.imwrite(os.path.join(save_path, f"env_{env_key}_cam_{cam}_step_{frame_id}.jpg"),
                                                rgb_images[f"{cam}-rgb"])
                        frame_id += 1

                        if done:
                            break
                        elif step == steps_per_obj - 1:
                            env.expert_phase = 0

                except Exception as e:
                    print(seed, env_key, model_id, e)
                    env.expert_phase = 0

            if idx == 0:
                cnt_list.append(frame_id)

            if seed < num_vid:
                for cam in cameras:
                    video_writer[cam].close()


    print(np.sum(cnt_list))
    pickle.dump(cnt_list, open(os.path.join(save_dir, f"cnt.pkl"), "wb"))

if __name__ == "__main__":
    collect_rand_and_cano_data()
