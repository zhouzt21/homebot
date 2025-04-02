
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio

import sys
sys.path.append("/home/zhouzhiting/Projects")

from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
import requests
from datetime import datetime

from homebot.homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv
from homebot.homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

import pickle
from diffusion_policy.policy import DiffusionPolicy
import torchvision.transforms as transforms

def eval_imitation_with_goal():
    """"
    Evaluate the imitation policy with goal. from pick_and_place_panda env.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        domain_randomize=True,
        canonical=True
    )

    goal_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        domain_randomize=True,
        canonical=True
    )

    cameras = ["third", "wrist"]
    usage = ["obs", "goal"]

    save_dir = "tmp"
    # save_dir = "try"
    # num_seeds = 10000
    num_eval = 10
    os.makedirs(save_dir, exist_ok=True)

    # cnt_list = []

    from tqdm import tqdm

    for i_eval in tqdm(range(num_eval)):
        seed = i_eval + 1000
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)
        goal_env.reset(seed=seed)

        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        # print(model_id_list)
        random_state.shuffle(model_id_list)

        # success_list = []

        for ep_id, model_id in enumerate(model_id_list):

            frame_id = 0
            success = False

            ep_path = os.path.join(save_path, f"ep_{ep_id}")
            os.makedirs(ep_path, exist_ok=True)
            goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
            goal_q_rand = random_state.uniform(-0.5, 0.5)

            prev_privileged_obs = None
            prev_obs = None
            for step in range(500):
                action, done = goal_env.expert_action(
                    noise_scale=0.0, obj_id=model_id,
                    goal_obj_pose=sapien.Pose(
                        p=np.concatenate([np.array([0.2, -0.2]) + goal_p_rand, [0.76]]),
                        q=euler2quat(np.array([0, 0, goal_q_rand]))
                    )
                )
                _, _, _, _, info = goal_env.step(action)

                # rgb_images = env.render_all()
                # rgb_images = env.capture_images_new(cameras=cameras)
                if step < 400:
                    if done:
                        p = goal_env.objs[model_id]["actor"].get_pose().p
                        if 0.05 < p[0] < 0.35 and -0.35 < p[1] < -0.05:
                            success = True
                        break
                    obs = goal_env.get_observation()
                    if prev_privileged_obs is not None and np.all(
                            np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                        goal_env.expert_phase = 0
                        break
                    prev_privileged_obs = obs["privileged_obs"]
                    prev_obs = obs
                    # print(obs.keys())
                    # data_item = {}
                    # for k, image in rgb_images.items():
                    #     if "third" in k:
                    #         data_item[k] = image

                    # for cam in cameras:
                    #     video_writer[cam].append_data(obs[f"{cam}-rgb"])
                    # if seed == 0 and idx == 0 and step == 0:
                    #     # print(data_item.keys())
                    #     imageio.imwrite(f'third-rgb-0.jpg', rgb_images["third-rgb"])

                    # data.append(data_item)
                    # data.append(rgb_images["third-rgb"])

                    # for cam in cameras:
                    #     imageio.imwrite(os.path.join(ep_path, f"cam_{cam}_step_{frame_id}.jpg"),
                    #                     obs[f"{cam}-rgb"])
                    # pickle.dump(obs, open(os.path.join(ep_path, f"step_{frame_id}.pkl"), "wb"))
                    frame_id += 1

                else:
                    if done:
                        break
                    goal_env.expert_phase = 6

            if success:
                # success_list.append((ep_id, "s", frame_id))
                print(seed, ep_id, "s", frame_id)
                video_writer = {cam: imageio.get_writer(
                    os.path.join(ep_path, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
                    # fps=40,
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

                for cam in cameras:
                    imageio.imwrite(os.path.join(ep_path, f"goal-{cam}.jpg"), prev_obs[f"{cam}-rgb"])

                for step in range(1000):
                    obs = env.get_observation()
                    for cam in cameras:
                        imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

                    files = {
                        f"{usg}-{cam}": open(os.path.join(ep_path, f"{usg}-{cam}.jpg"), "rb")
                        for cam in cameras for usg in usage
                    }
                    response = requests.post("http://localhost:9977/diffusion", files=files)
                    response = response.json()

                    action = np.array(response["action"])
                    delta_pos = action[:3]
                    gripper_width = action[-1]

                    mat_6 = action[3:9].reshape(3, 2)
                    mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
                    mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
                    z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
                    mat = np.c_[mat_6, z_vec]
                    assert mat.shape == (3, 3)

                    delta_euler = (
                            np.clip(
                                wrap_to_pi(quat2euler(mat2quat(mat))) / env.rot_scale,
                                -1.0,
                                1.0,
                            )
                            * env.rot_scale
                    )

                    action = np.concatenate(
                        [
                            [0, 0],
                            delta_pos,
                            delta_euler,
                            [gripper_width],
                        ]
                    )

                    _, _, _, _, info = env.step(action)

                for writer in video_writer.values():
                    writer.close()

                exit()

            else:
                # success_list.append((ep_id, "f", frame_id))
                print(seed, ep_id, "f", frame_id)

                # if done:
                # print(f"{model_id} done! use step {step}.")
                # video_writer.close()
                # p = envs[0].objs[model_id]["actor"].get_pose().p
                # if 0.1 < p[0] < 0.4 and -0.35 < p[1] < -0.05:
                #     success_list.append(model_id)
                # else:
                #     print(model_id, p)
                # break
                # elif step == 399:
                # print(f"{model_id} step used out. ")
                # pass

            # print(len(frames))
            # if idx == 0:
            #     cnt_list.append(frame_id)
            # cnt_list.append(len(data))

            # pickle.dump(data, open(os.path.join(save_dir, f"seed_{seed}_env_{idx}.pkl"), "wb"))

            # env = deepcopy(goal_env)

        # print(success_list)
        # pickle.dump(success_list, open(os.path.join(save_path, f"info.pkl"), "wb"))


#######  init_policy ########
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

###############################


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
        output: image_data: (B, c, h, w),  normalized in [0,1], float, np
                qpos_data: (10)
    """
    # all_cam_images = np.stack(image_list, axis=0)   # already in shape (B, h, w, c)
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
        Evaluate the cano policy with sim2sim, from pick_and_place_panda random env.
        Comunicate with the ldm server.
        loop: num_eval
            loop: num_obj
                loop: num_pred
                    step action 10 steps for each predict
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical= False, #True,
        action_relative="none",
        allow_dir=["along"]
    )

    cameras = ["third"] #, "wrist"]
    usage = ["obs"]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join("tmp", stamp)

    num_eval = 10
    os.makedirs(save_dir, exist_ok=True)

    from tqdm import tqdm

    for i_eval in range(num_eval):
        seed = i_eval + 1000
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        random_state.shuffle(model_id_list)

        for ep_id, model_id in enumerate(model_id_list):

            ep_path = os.path.join(save_path, f"ep_{ep_id}")
            os.makedirs(ep_path, exist_ok=True)

            video_writer = {cam: imageio.get_writer(
                os.path.join(ep_path, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            # for step in tqdm(range(500)):
            num_pred = 30
            for step in tqdm(range(num_pred)):
                obs = env.get_observation()
                for cam in cameras:
                    imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                    video_writer[cam].append_data(obs[f"{cam}-rgb"])

                files = {
                    f"{usg}-{cam}": open(os.path.join(ep_path, f"{usg}-{cam}.jpg"), "rb")
                    for cam in cameras for usg in usage
                }

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

                ##### for eval   
                response = requests.post("http://localhost:9977/ldm", files=files)
                response_data = response.json()

                all_cam_images = np.array(response_data["samples"])
                print(all_cam_images.shape)
                image_data, qpos_data = process_data(all_cam_images, proprio_state)
                image_data, qpos_data = image_data.cuda().unsqueeze(0), qpos_data.cuda().unsqueeze(0)
                print(image_data.shape, qpos_data.shape)

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
                            init_to_desired_pose[:3, 3],
                            mat2euler(init_to_desired_pose[:3, :3]),
                            [gripper_width]
                        ]
                    )
                    converted_actions.append(pose_action)

                    _, _, _, _, info = env.step(pose_action)

                    for cam in cameras:
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

            for writer in video_writer.values():
                writer.close()


if __name__ == "__main__":
    eval_imitation()
    # eval_imitation_with_goal()