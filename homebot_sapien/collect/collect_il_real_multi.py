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

from homebot_sapien.vec_wrapper.subproc_vec_env import SubprocVecEnv

PANDA_DATA = "/home/zhouzhiting/Data/panda_data"

def convert_pose_to_array(pose: sapien.Pose):
    """
    将 sapien.Pose 转换为 (7,) 数组：前三个为位置信息，后四个为四元数表示旋转
    """
    return np.concatenate([pose.p, pose.q])


def make_env(device, seed, index, canonical=True):
    """创建单个环境的函数工厂"""
    def _init():
        env = PickAndPlaceEnv(
            use_gui=False,
            device=device,
            obs_keys=(),
            domain_randomize=True,
            canonical=canonical,
        )
        env.reset(seed=seed)
        return env
    return _init

def collect_imitation_data_multi(n_envs=4):
    """
    多进程收集模仿学习数据
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置参数
    save_dir = os.path.join(PANDA_DATA, "cano_1")
    num_seeds = 5000
    num_vid = 20
    cameras = ["third"]
    os.makedirs(save_dir, exist_ok=True)
    num_suc = 0
    
    from tqdm import tqdm
    
    # 按批次处理种子
    for batch_start in tqdm(range(0, num_seeds, n_envs)):
        batch_end = min(batch_start + n_envs, num_seeds)
        batch_seeds = list(range(batch_start, batch_end))
        env_count = len(batch_seeds)
        
        # 创建环境函数列表
        env_fns = [make_env(device, seed, i) for i, seed in enumerate(batch_seeds)]
        
        # 创建向量化环境
        vec_env = SubprocVecEnv(env_fns, reset_when_done=False)
        
        # 为每个种子创建保存目录
        for seed in batch_seeds:
            save_path = os.path.join(save_dir, f"seed_{seed}")
            os.makedirs(save_path, exist_ok=True)
        
        # 初始化随机状态
        random_states = [np.random.RandomState(seed=seed) for seed in batch_seeds]
        
        # 获取模型ID列表 - 对每个环境分别获取
        model_id_lists = []
        for i in range(env_count):
            model_ids = vec_env.get_attr("objs", indices=[i])[0].keys()
            model_ids = list(model_ids)
            random_states[i].shuffle(model_ids)
            model_id_lists.append(model_ids)
        
        # 为每个环境跟踪成功列表
        success_lists = [[] for _ in range(env_count)]
        
        # 对每个环境的每个模型ID循环处理
        for i, (seed, model_ids) in enumerate(zip(batch_seeds, model_id_lists)):
            save_path = os.path.join(save_dir, f"seed_{seed}")
            
            for ep_id, model_id in enumerate(model_ids):
                # 为视频创建写入器
                if seed < num_vid:
                    os.makedirs("tmp", exist_ok=True)
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
                    goal_p_rand = random_states[i].uniform(-0.1, 0.1, size=(2,))
                    goal_q_rand = random_states[i].uniform(-0.5, 0.5)
                    
                    goal_obj_pose = sapien.Pose(
                        p=np.concatenate([np.array([0.4, -0.2]) + goal_p_rand, [0.76]]),
                        q=euler2quat(np.array([0, 0, goal_q_rand]))
                    )
                    
                    for step in range(500):
                        # 获取专家动作
                        result = vec_env.env_method(
                            "expert_action", 
                            noise_scale=0.2, 
                            obj_id=model_id,
                            goal_obj_pose=goal_obj_pose,
                            indices=[i]
                        )[0]
                        
                        action, done_expert, desired_dict = result
                        
                        # 创建全部环境的动作列表
                        all_actions = [np.zeros(7) for _ in range(env_count)]  # 假设动作维度为7
                        all_actions[i] = action
                        
                        # 执行步骤
                        _, _, dones, truncated, infos = vec_env.step(all_actions)
                        done = dones[i]
                        
                        if step < 400:
                            if done:
                                # 检查成功条件
                                p = vec_env.env_method("get_attr", 
                                                      f"objs['{model_id}']['actor'].get_pose().p", 
                                                      indices=[i])[0]
                                if 0.25 < p[0] < 0.55 and -0.35 < p[1] < -0.05:
                                    success = True
                                    num_suc += 1
                                break
                            
                            # 获取观察
                            obs = vec_env.env_method("get_observation", indices=[i])[0]
                            
                            # 获取TCP姿态
                            tcp_pose = vec_env.env_method("_get_tcp_pose", indices=[i])[0]
                            episode_data['tcp_pose'].append(convert_pose_to_array(tcp_pose))
                            
                            # 获取其他数据
                            episode_data['gripper_width'].append(vec_env.env_method("_get_gripper_width", indices=[i])[0])
                            episode_data['robot_joints'].append(vec_env.env_method("get_attr", "robot.get_qpos()", indices=[i])[0].copy())
                            episode_data['privileged_obs'].append(obs["privileged_obs"].copy())
                            episode_data['action'].append(action.copy())
                            episode_data['desired_grasp_pose'].append(convert_pose_to_array(desired_dict["desired_grasp_pose"]))
                            episode_data['desired_gripper_width'].append(desired_dict["desired_gripper_width"])
                            
                            # 保存图像
                            for cam in cameras:
                                image = obs.pop(f"{cam}-rgb")
                                imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{cam}.jpg"), image)
                                if seed < num_vid:
                                    video_writer[cam].append_data(image)
                            
                            frame_id += 1
                        else:
                            if done:
                                break
                            # 设置专家阶段
                            vec_env.env_method("set_attr", "expert_phase", 6, indices=[i])
                            
                except Exception as e:
                    print(f"Seed {seed}, ep {ep_id}: {e}")
                
                if success:
                    success_lists[i].append((ep_id, "s", frame_id))
                else:
                    success_lists[i].append((ep_id, "f", frame_id))
                
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
            
            # 保存成功列表
            pickle.dump(success_lists[i], open(os.path.join(save_path, f"info.pkl"), "wb"))
            
        # 关闭环境
        vec_env.close()
    
    print(f"Total successes: {num_suc}")

if __name__ == "__main__":
    collect_imitation_data_multi(n_envs=4)