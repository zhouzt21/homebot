import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
import sys
sys.path.append('/home/zhouzhiting/Projects/homebot')

from homebot_sapien.utils.math import wrap_to_pi, euler2quat
from homebot_sapien.env.pick_and_place_panda_real import PickAndPlaceEnv

def eval_il_expert():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        allow_dir=["along"] #, "column", "side"
    )

    env = cano_pick_env
    cameras = ["third"]

    num_seeds = 10
    num_vid = 20
    num_suc = 0

    from tqdm import tqdm

    total_success_list = []

    for seed in tqdm(range(num_seeds)):
        ### tmp change
        seed = seed +10 
        env.reset(seed=seed)
        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        random_state.shuffle(model_id_list)

        success_list = []

        for ep_id, model_id in enumerate(model_id_list):
            save_path = os.path.join("tmp", f"{model_id[1]}")
            print("actual", model_id)
            os.makedirs(save_path, exist_ok=True)
            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"tmp/{model_id[1]}/seed_{seed}.mp4",
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            success = False
            frame_id = 0

            try:
                goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))   ### can be changed
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

                        for cam in cameras:
                            image = obs.pop(f"{cam}-rgb")
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
                total_success_list.append(("s", seed))
                num_suc += 1
            else:
                success_list.append((ep_id, "f", frame_id))

            if seed < num_vid:
                for writer in video_writer.values():
                    writer.close()

    print("success rate", num_suc / num_seeds)
    print("num_suc", num_suc)
    print("total_success_list", total_success_list)
    

if __name__ == "__main__":
    eval_il_expert()