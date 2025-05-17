import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import pickle 
import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')

from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos
from homebot_sapien.utils.wrapper import StateObservationWrapper, TimeLimit
from homebot_sapien.utils.math import euler2quat
###### need to be set ######
from homebot_sapien.env.drawer import PushAndPullEnv
## support drawer, microwave

def convert_pose_to_array(pose: sapien.Pose):
    """
    将 sapien.Pose 转换为 (7,) 数组：前三个为位置信息，后四个为四元数表示旋转
    """
    return np.concatenate([pose.p, pose.q])


def collect_imitation_data():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PushAndPullEnv(
        use_gui=False,
        device=device,
        obs_keys=("tcp_pose", "gripper_width"),
        domain_randomize=True,
        canonical=True,
    )
    env_wrapper = StateObservationWrapper(TimeLimit(env))

    cameras = ["third"]

    ##### need to be set #######
    save_dir = "/home/zhouzhiting/Data/panda_data/single_obj/drawer"
    name = "drawer"
    num_seeds = 1000
    num_vid = 10
    ###########################

    os.makedirs(save_dir, exist_ok=True)
    num_suc = 0
    success_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)
        
        ep_path = os.path.join(save_path, "ep_0")
        os.makedirs(ep_path, exist_ok=True)

        env_wrapper.env.reset(seed=seed)

        if seed < num_vid:
            video_path = f"tmp/collect_il/{name}"
            os.makedirs(video_path, exist_ok=True)
            video_writer = {cam: imageio.get_writer(
                f"{video_path}/seed_{seed}.mp4",
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
            prev_privileged_obs = None
            while True:
                action, done, desired_dict = env_wrapper.env.expert_action(
                    noise_scale=0.2,
                )

                o, _, _, _, info = env_wrapper.env.step(action)
                o = env_wrapper.process_obs(o)

                if frame_id < 500:
                    if done:
                        success = True
                        break
                        
                    obs = env_wrapper.env.get_observation()

                    for cam in cameras:
                        image = obs.pop(f"{cam}-rgb")
                        imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{cam}.jpg"), image) 
                        if seed < num_vid:
                            video_writer[cam].append_data(image)
                    
                    obs.update({"action": action})
                    obs.update(desired_dict)
                    obs.update({"wrapper_obs": o})
                    
                    if prev_privileged_obs is not None and np.all(
                            np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                        env_wrapper.env.expert_phase = 0
                        break
                    prev_privileged_obs = obs["privileged_obs"]
                    
                    episode_data['tcp_pose'].append(obs["tcp_pose"].copy())  
                    episode_data['gripper_width'].append(obs["gripper_width"].copy())
                    # episode_data['robot_joints'].append(obs["robot_joints"].copy()) 
                    episode_data['privileged_obs'].append(obs["privileged_obs"].copy())
                    episode_data['action'].append(obs["action"].copy())
                    episode_data['desired_grasp_pose'].append(convert_pose_to_array(desired_dict["desired_grasp_pose"]))
                    episode_data['desired_gripper_width'].append(desired_dict["desired_gripper_width"])

                    frame_id += 1
                else:
                    break

        except Exception as e:
            print("error: ", seed, e)
        
        try:
            if len(episode_data['tcp_pose']) > 0:
                episode_array = {
                    'tcp_pose': np.stack(episode_data['tcp_pose']),
                    'gripper_width': np.array(episode_data['gripper_width']),
                    # 'robot_joints': np.stack(episode_data['robot_joints']),
                    'privileged_obs': np.stack(episode_data['privileged_obs']),
                    'action': np.stack(episode_data['action']),
                    'desired_grasp_pose': np.stack(episode_data['desired_grasp_pose']),
                    'desired_gripper_width': np.array(episode_data['desired_gripper_width'])
                }
                np.savez(os.path.join(ep_path, "total_steps.npz"), **episode_array)
        except Exception as e:
            print(f"Error saving episode data for seed {seed}: {e}")

        if success:
            success_list.append((0, "s", frame_id))  # 使用ep_id=0
            num_suc += 1
        else:
            success_list.append((0, "f", frame_id))  # 使用ep_id=0

        if seed < num_vid:
            for writer in video_writer.values():
                writer.close()

    pickle.dump(success_list, open(os.path.join(save_dir, f"info.pkl"), "wb"))

    print(num_suc)
    
if __name__ == "__main__":
    collect_imitation_data()