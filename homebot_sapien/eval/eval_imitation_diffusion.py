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
from homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv
from diffusion_policy.policy import DiffusionPolicy


################## init policy ##################
# in docker
# OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/root/data/cano_policy_1/", "norm_stats.pkl"), "rb"))
# OBS_NORMALIZE_PARAMS = pickle.load(open(os.path.join("/root/data/cano_drawer_0915/", "norm_stats_1.pkl"), "rb"))
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
# ckpt_path = "/root/data/diffusion_policy_checkpoints/20240415_083018/policy_step_200000_seed_0.ckpt"  # two obs xarm
# ckpt_path = "/root/data/diffusion_policy_checkpoints/20240915_133854/policy_step_500000_seed_0.ckpt"
# ckpt_path = "/home/zhouzhiting/panda_data/diffusion_policy_checkpoints/20250228_211805/policy_step_2000_seed_0.ckpt"
# ckpt_path = "/home/zhouzhiting/panda_data/diffusion_policy_checkpoints/20250303_115710/policy_step_15000_seed_0.ckpt"
ckpt_path = "/home/zhouzhiting/Data/panda_data/diffusion_policy_checkpoints/20250307_191102/policy_step_250000_seed_0.ckpt"


policy = DiffusionPolicy(policy_config)
policy.deserialize(torch.load(ckpt_path))
policy.eval()
policy.cuda()

##################################

def process_action(action):
    """
        unnormalize the action to the original scale, prepare for the env step.
        input: action: (10,), float, np
        output: action: (10,), float, np
    """
    action = action * np.expand_dims(pose_gripper_scale, axis=0) + np.expand_dims(pose_gripper_mean, axis=0)
    return action

def process_data(image_list, proprio_state):
    """
        process the data for diffusion policy model. 
        input:  image_list: [ M * (h, w, c)],  uint8, np (M is the number of cameras)
                proprio_state: (10,), float, np
        output: image_data: (M, c, h, w),  normalized in [0,1], float, np
                qpos_data: (10)
    """
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
    """"
        evaluate the cano policy in the pick and place env.
        loop: num_eval
            loop: num_obj
                loop: num_pred
                    step action 10 steps for each prediction
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="none"
    )

    # cameras = ["third", "wrist"]
    # usage = ["obs"]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join("tmp", stamp)

    num_eval = 10
    os.makedirs(save_dir, exist_ok=True)

    from tqdm import tqdm

    for i_eval in range(num_eval):
        seed = i_eval + 1000
        # seed = 4
        # save_path = os.path.join(save_dir, f"seed_{seed}")
        # os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        # random_state = np.random.RandomState(seed=seed)

        # model_id_list = list(env.objs.keys())
        # print(model_id_list)
        # random_state.shuffle(model_id_list)
        model_id_list = [None]

        for ep_id, model_id in enumerate(model_id_list):

            # ep_path = os.path.join(save_path, f"ep_{ep_id}")
            # os.makedirs(ep_path, exist_ok=True)

            video_writer = {cam: imageio.get_writer(
                os.path.join(save_dir, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            num_pred = 40
            for _ in tqdm(range(num_pred)):
                obs = env.get_observation()
                image_list = []
                for cam in cameras:
                    image_list.append(obs[f"{cam}-rgb"])
                    # imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                    # video_writer[cam].append_data(obs[f"{cam}-rgb"])

                pose = obs["tcp_pose"]
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
                        np.array([obs["gripper_width"]]),
                    ]
                )

                image_data, qpos_data = process_data(image_list, proprio_state)
                image_data, qpos_data = image_data.cuda().unsqueeze(0), qpos_data.cuda().unsqueeze(0)
                # print(image_data.shape, qpos_data.shape)

                pred_actions = policy(qpos_data, image_data).squeeze().cpu()
                actions = process_action(pred_actions)

                converted_actions = []

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
                    pose_action = np.concatenate(
                        [
                            # [0, 0], # for mobile base
                            init_to_desired_pose[:3, 3],
                            mat2euler(init_to_desired_pose[:3, :3]),
                            [gripper_width]
                        ]
                    )
                    converted_actions.append(pose_action)

                    _, _, _, _, info = env.step(pose_action)

                    obs = env.get_observation()
                    for cam in cameras:
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

            for writer in video_writer.values():
                writer.close()


if __name__ == "__main__":
    eval_imitation()
