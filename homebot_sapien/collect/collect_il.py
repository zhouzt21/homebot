
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import pickle 
import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')

from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos
from homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv
# from Projects.homebot.config import PANDA_DATA
PANDA_DATA = "/home/zhouzhiting/panda_data"



def collect_imitation_data():
    """
    Collect imitation data for the policy, from pick_and_place_panda env.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        # canonical=False,
        allow_dir=["along"]
    )

    env = cano_pick_env
    cameras = ["third"]

    # save_dir = "/root/data/rand_policy_pd_1"
    # save_dir = os.path.join(PANDA_DATA, "rand_policy_pd_1")
    save_dir = os.path.join(PANDA_DATA, "cano_policy_pd_2")  # 500 traj for test

    # save_dir = "try"
    # num_seeds = 5000
    num_seeds = 5000  # cano test
    # num_seeds = 10
    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    # cnt_list = []
    num_suc = 0

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        # env.scene.set_timestep(0.01)
        # env.frame_skip = 5

        random_state = np.random.RandomState(seed=seed)

        # data = []

        model_id_list = list(env.objs.keys())
        # print(model_id_list)
        random_state.shuffle(model_id_list)

        success_list = []

        for ep_id, model_id in enumerate(model_id_list):

            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"tmp/seed_{seed}_ep_{ep_id}_cam_{cam}.mp4",
                    # fps=40,
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            success = False
            frame_id = 0

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

                    # rgb_images = env.render_all()
                    # rgb_images = env.capture_images_new(cameras=cameras)
                    if step < 400:
                        if done:
                            p = env.objs[model_id]["actor"].get_pose().p
                            if 0.25 < p[0] < 0.55 and -0.35 < p[1] < -0.05:
                                success = True
                            break
                        obs = env.get_observation()
                        obs.update({"action": action})
                        obs.update(desired_dict)
                        if prev_privileged_obs is not None and np.all(
                                np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                            env.expert_phase = 0
                            break
                        prev_privileged_obs = obs["privileged_obs"]
                        # print(obs.keys())
                        # data_item = {}
                        # for k, image in rgb_images.items():
                        #     if "third" in k:
                        #         data_item[k] = image

                        # if seed < num_vid:
                        #     for cam in cameras:
                        #         video_writer[cam].append_data(obs[f"{cam}-rgb"])
                        # if seed == 0 and idx == 0 and step == 0:
                        #     # print(data_item.keys())
                        #     imageio.imwrite(f'third-rgb-0.jpg', rgb_images["third-rgb"])

                        # data.append(data_item)
                        # data.append(rgb_images["third-rgb"])

                        # for cam in cameras:
                        #     imageio.imwrite(os.path.join(ep_path, f"cam_{cam}_step_{frame_id}.jpg"),
                        #                     obs[f"{cam}-rgb"])
                        for cam in cameras:
                            image = obs.pop(f"{cam}-rgb")
                            imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{cam}.jpg"), image)
                            if seed < num_vid:
                                video_writer[cam].append_data(image)

                        pickle.dump(obs, open(os.path.join(ep_path, f"step_{frame_id}.pkl"), "wb"))
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
                # print(seed, ep_id, "s", frame_id)
            else:
                success_list.append((ep_id, "f", frame_id))
                # print(seed, ep_id, "f", frame_id)

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

            if seed < num_vid:
                for writer in video_writer.values():
                    writer.close()

        pickle.dump(success_list, open(os.path.join(save_path, f"info.pkl"), "wb"))

    print(num_suc)

if __name__ == "__main__":
    collect_imitation_data()

