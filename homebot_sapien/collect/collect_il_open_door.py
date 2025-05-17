import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import pickle
import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')

from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos
from homebot_sapien.env.open_door import OpenDoorEnv

def convert_pose_to_array(pose: sapien.Pose):
    """
    将 sapien.Pose 转换为 (7,) 数组：前三个为位置信息，后四个为四元数表示旋转
    """
    return np.concatenate([pose.p, pose.q])

def collect_door_data():
    """
    收集开门任务的数据，格式与collect_il_real.py类似
    数据格式：
        'tcp_pose': np.stack(episode_data['tcp_pose']),              # (N, 7)
        'gripper_width': np.array(episode_data['gripper_width']),    # (N,)
        'action': np.stack(episode_data['action']),                  # (N, action_dim)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #### need to be set ######
    PANDA_DATA = "/home/zhouzhiting/Data/panda_data"
    save_dir = os.path.join(PANDA_DATA, "open_door_data")    
    name = "door"
    mode = "third"  ## cannot be "wrist" now
    num_seeds = 1000
    num_vid = 20
    #########################

    num_suc = 0
    os.makedirs(save_dir, exist_ok=True)
    
    from tqdm import tqdm
    
    try:
        for seed in tqdm(range(num_seeds)):
            save_path = os.path.join(save_dir, f"seed_{seed}")
            os.makedirs(save_path, exist_ok=True)
            
            open_door_env = OpenDoorEnv(
                use_gui=False,
                device=device,
                obs_keys=(f"{mode}-rgb", "tcp_pose", "gripper_width"),
                door_from_urdf=False,
                domain_randomize=True,  # 
            )
            
            open_door_env.reset(seed=seed)
            
            ep_path = os.path.join(save_path, "ep_0")
            os.makedirs(ep_path, exist_ok=True)
            
            if seed < num_vid:
                video_path = os.path.join(f"tmp/{name}", "videos")
                os.makedirs(video_path, exist_ok=True)
                video_writer = imageio.get_writer(
                    f"{video_path}/seed_{seed}_{name}.mp4",
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                )
            
            success = False
            frame_id = 0
            
            episode_data = {
                'tcp_pose': [],
                'gripper_width': [],
                'action': [],
            }
            
            try:
                for step in range(600):  
                    action = open_door_env.expert_action(noise_scale=0.5)
                    _, _, _, _, info = open_door_env.step(action)
                    
                    obs = open_door_env.get_observation()
                    
                    # episode_data['tcp_pose'].append(convert_pose_to_array(env._get_tcp_pose()))  ## ??
                    # episode_data['gripper_width'].append(env._get_gripper_width())  ## ??
                    episode_data['tcp_pose'].append(obs["tcp_pose"].copy())  
                    episode_data['gripper_width'].append(obs["gripper_width"].copy())  
                    episode_data['robot_joints'].append(obs["robot_joints"].copy())  
                    episode_data['privileged_obs'].append(obs["privileged_obs"].copy())
                    episode_data['action'].append(action.copy())
                    
                    rgb_image = open_door_env.render()
                    imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{mode}.jpg"), rgb_image)
                    video_writer.append_data(rgb_image)
                    
                    frame_id += 1
                    
                    done = info["is_success"] or step >= 599
                    if done:
                        if info["is_success"]:
                            success = True
                            num_suc += 1
                        break
                        
            except Exception as e:
                print(f"Error during episode collection for seed {seed}: {e}")
            
            if seed < num_vid:
                video_writer.close()

            try:
                episode_array = {
                    'tcp_pose': np.stack(episode_data['tcp_pose']),
                    'gripper_width': np.array(episode_data['gripper_width']),
                    'robot_joints': np.stack(episode_data['robot_joints']),
                    'privileged_obs': np.stack(episode_data['privileged_obs']),
                    'action': np.stack(episode_data['action']),
                }
                np.savez(os.path.join(ep_path, "total_steps.npz"), **episode_array)
                
                success_info = [(0, "s" if success else "f", frame_id)]
                pickle.dump(success_info, open(os.path.join(save_path, "info.pkl"), "wb"))
            except Exception as e:
                print(f"Error saving episode data for seed {seed}: {e}")
            
            open_door_env.close()
    
    except KeyboardInterrupt:
        print("Ctrl+C detected, exiting.")
    finally:
        print(f"Successfully collected {num_suc} episodes out of {min(seed+1, num_seeds)}")

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_filename = "test"
    video_writer = imageio.get_writer(
        f"{video_filename}.mp4",
        fps=40,
        format="FFMPEG",
        codec="h264",
    )
    open_door_sim = OpenDoorEnv(
        use_gui=False,
        device=device,
        obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        door_from_urdf=False,
        # use_real=True,
        # domain_randomize=False,
    )
    open_door_sim.reset()
    step_count = 0
    traj_count = 0
    success_count = 0
    done = False
    while traj_count < 10:
        action = open_door_sim.expert_action(noise_scale=0.5)
        # exit()
        _, _, _, _, info = open_door_sim.step(action)
        step_count += 1
        rgb_image = open_door_sim.render()
        video_writer.append_data(rgb_image)
        done = info["is_success"] or step_count >= 600
        if done:
            traj_count += 1
            step_count = 0
            success_count += info["is_success"]
            open_door_sim.reset()
    print("success rate", success_count / traj_count)
    video_writer.close()


if __name__ == "__main__":
    collect_door_data()