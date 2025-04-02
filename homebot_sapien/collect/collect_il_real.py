import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import pickle 
import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')

from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos
from homebot_sapien.env.pick_and_place_panda_real import PickAndPlaceEnv

PANDA_DATA = "/home/zhouzhiting/Data/panda_data"

def convert_pose_to_array(pose: sapien.Pose):
    """
    将 sapien.Pose 转换为 (7,) 数组：前三个为位置信息，后四个为四元数表示旋转
    """
    return np.concatenate([pose.p, pose.q])

def collect_imitation_data():
    """
    除了图像之外，其它数据全部存到一个pkl文件中
    数据格式：
        'tcp_pose': np.stack(episode_data['tcp_pose']),              # (N, 7)
        'gripper_width': np.array(episode_data['gripper_width']),      # (N,)
        'robot_joints': np.stack(episode_data['robot_joints']),        # (N, num_joints)
        'privileged_obs': np.stack(episode_data['privileged_obs']),    # (N, obs_dim)
        'action': np.stack(episode_data['action']),                    # (N, action_dim)
        'desired_grasp_pose': np.stack(episode_data['desired_grasp_pose']),  # (N, 7)
        'desired_gripper_width': np.array(episode_data['desired_gripper_width'])  # (N,)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        allow_dir=["along", "column"]
    )

    env = cano_pick_env
    cameras = ["third"]

    save_dir = os.path.join(PANDA_DATA, "cano_policy_pd_3")  # 例如500 traj
    num_seeds = 5000
    num_vid = 20
    os.makedirs(save_dir, exist_ok=True)
    num_suc = 0

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)
        env.reset(seed=seed)
        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        random_state.shuffle(model_id_list)

        success_list = []

        for ep_id, model_id in enumerate(model_id_list):
            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"tmp/seed_{seed}_ep_{ep_id}_cam_{cam}.mp4",
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            success = False
            frame_id = 0

            episode_data = {
                'tcp_pose': [],
                'gripper_width': [],
                'robot_joints': [],
                'privileged_obs': [],
                'action': [],
                'desired_grasp_pose': [],
                'desired_gripper_width': []
            }
            try:
                ep_path = os.path.join(save_path, f"ep_{ep_id}")
                os.makedirs(ep_path, exist_ok=True)
                goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
                goal_q_rand = random_state.uniform(-0.5, 0.5)

                prev_privileged_obs = None
                for step in range(500):
                    action, done, desired_dict = env.expert_action(
                        noise_scale=0.2, obj_id=model_id,
                        goal_obj_pose=sapien.Pose(
                            p=np.concatenate([np.array([0.4, -0.2]) + goal_p_rand, [0.76]]),
                            q=euler2quat(np.array([0, 0, goal_q_rand]))
                        )
                    )
                    _, _, _, _, info = env.step(action)

                    if step < 400:
                        if done:
                            p = env.objs[model_id]["actor"].get_pose().p
                            # offset_x, offset_y, offset_z = env.objs[model_id]["offset"]
                            # print( offset_x, offset_y)
                            if 0.25 < p[0] < 0.55 and -0.35 < p[1]< -0.05:
                                success = True
                            break

                        obs = env.get_observation()

                        episode_data['tcp_pose'].append(convert_pose_to_array(env._get_tcp_pose()))
                        episode_data['gripper_width'].append(env._get_gripper_width())
                        episode_data['robot_joints'].append(env.robot.get_qpos().copy())
                        episode_data['privileged_obs'].append(obs["privileged_obs"].copy())
                        episode_data['action'].append(action.copy())

                        episode_data['desired_grasp_pose'].append(convert_pose_to_array(desired_dict["desired_grasp_pose"]))
                        episode_data['desired_gripper_width'].append(desired_dict["desired_gripper_width"])

                        for cam in cameras:
                            image = obs.pop(f"{cam}-rgb")
                            imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{cam}.jpg"), image)
                            if seed < num_vid:
                                video_writer[cam].append_data(image)

                        frame_id += 1
                    else:
                        if done:
                            break
                        env.expert_phase = 6
            except Exception as e:
                print(seed, ep_id, e)

            if success:
                success_list.append((ep_id, "s", frame_id))
                num_suc += 1
            else:
                success_list.append((ep_id, "f", frame_id))

            try:
                episode_array = {
                    'tcp_pose': np.stack(episode_data['tcp_pose']),
                    'gripper_width': np.array(episode_data['gripper_width']),
                    'robot_joints': np.stack(episode_data['robot_joints']),
                    'privileged_obs': np.stack(episode_data['privileged_obs']),
                    'action': np.stack(episode_data['action']),
                    'desired_grasp_pose': np.stack(episode_data['desired_grasp_pose']),
                    'desired_gripper_width': np.array(episode_data['desired_gripper_width'])
                }
                pickle.dump(episode_array, open(os.path.join(ep_path, f"total_steps.pkl"), "wb"))
            except Exception as e:
                print(f"Error saving episode data for seed {seed}, ep {ep_id}: {e}")

            if seed < num_vid:
                for writer in video_writer.values():
                    writer.close()

        pickle.dump(success_list, open(os.path.join(save_path, f"info.pkl"), "wb"))

    print(num_suc)

if __name__ == "__main__":
    collect_imitation_data()