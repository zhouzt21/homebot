# import json
# import numpy as np
#
# path = "./asset/mani_skill2_egad/info_pick_train_v0.json"
# data = json.load(open(path, "r"))
#
# print(len(data))
# new_data = {}
#
# cnt = 0
#
# for obj_id, obj_dict in data.items():
#     new_obj_dict = obj_dict.copy()
#     bbox = obj_dict["bbox"]
#     bbmin, bbmax = np.array(bbox["min"]), np.array(bbox["max"])
#
#     bb = (bbmax - bbmin) * obj_dict["scales"]
#
#     if bb[-1] > 0.1:
#         continue
#
#     if max(bb[0], bb[1]) > 0.2:
#         continue
#
#     if min(bb[0], bb[1]) > 0.08:
#         continue
#
#     if bb[0] > bb[1]:
#         cnt += 1
#         along = "x"
#     else:
#         along = "y"
#
#     new_obj_dict.update({"along": along})
#     new_data.update({obj_id: new_obj_dict})
#
# print(len(new_data.keys()))
# print(cnt)
#
# json.dump(new_data, open("./asset/mani_skill2_egad/info_pick_train_v1.json", "w"), indent=2)


# path = "./asset/mani_skill2_ycb/info_pick_v1.json"
# data = json.load(open(path, "r"))
# success_list = json.load(open("success_model_id_2.json", "r"))
#
# print(len(data))
# print(len(success_list))
# new_data = {}
#
#
# for obj_id, obj_dict in data.items():
#     if obj_id in success_list:
#         new_obj_dict = obj_dict.copy()
#
#         new_data.update({obj_id: new_obj_dict})
#
# print(len(new_data.keys()))
#
# json.dump(new_data, open("./asset/mani_skill2_ycb/info_pick_v2.json", "w"), indent=2)

# import pickle
# from tqdm import tqdm
# import os
# import imageio
# import cv2
#
# cnt = []
# save_dir = f"/root/data/sim2sim_1"
# out_dir = "tmp"
#
# for seed in range(5):
#     seed_dir = os.path.join(save_dir, f"seed_{seed}")
#     img_path = os.listdir(seed_dir)
#     num_frame = len(img_path) // 2
#
#     rand_list = []
#     cano_list = []
#
#     for step in range(5):
#         rand_path = os.path.join(seed_dir, f"env_rand_step_{num_frame * step // 5 * 10}.jpg")
#         cano_path = os.path.join(seed_dir, f"env_cano_step_{num_frame * step // 5 * 10}.jpg")
#         rand_img = imageio.imread(rand_path)
#         cano_img = imageio.imread(cano_path)
#         rand_img = cv2.resize(rand_img, (240, 240))
#         cano_img = cv2.resize(cano_img, (240, 240))
#         rand_list.append(rand_img)
#         cano_list.append(cano_img)
#         # cat_img = np.concatenate([rand_img, cano_img], axis=1)
#
#     rand_cat = np.concatenate(rand_list, axis=1)
#     cano_cat = np.concatenate(cano_list, axis=1)
#     imageio.imwrite(os.path.join(out_dir, f"seed_{seed}_rand.jpg"), rand_cat)
#     imageio.imwrite(os.path.join(out_dir, f"seed_{seed}_cano.jpg"), cano_cat)
#
# exit()
#
# for seed in tqdm(range(10)):
#
#     video_writer = imageio.get_writer(
#         f"cat_seed_{seed}.mp4",
#         # fps=40,
#         fps=10,
#         format="FFMPEG",
#         codec="h264",
#     )
#
#     seed_dir = os.path.join(save_dir, f"seed_{seed}")
#     img_path = os.listdir(seed_dir)
#     num_frame = len(img_path) // 2
#
#     for step in range(num_frame):
#         rand_path = os.path.join(seed_dir, f"env_rand_step_{step * 10}.jpg")
#         cano_path = os.path.join(seed_dir, f"env_cano_step_{step * 10}.jpg")
#         rand_img = imageio.imread(rand_path)
#         cano_img = imageio.imread(cano_path)
#         cat_img = np.concatenate([rand_img, cano_img], axis=1)
#         video_writer.append_data(cat_img)
#
#     video_writer.close()
#
#     # for idx in [0, 1]:
#     #     path = os.path.join(save_dir, f"seed_{seed}_env_{idx}.pkl")
#     #     data = pickle.load(open(path, "rb"))
#     #     for step, image in enumerate(data):
#     #         save_path = os.path.join(save_dir, f"seed_{seed}")
#     #         os.makedirs(save_path, exist_ok=True)
#     #         imageio.imwrite(os.path.join(save_path, f"env_{idx}_step_{step}.jpg"), image)
#     #     if idx:
#     #         cnt.append(len(data))
#
# print(len(cnt))
# print(np.sum(cnt))
# # pickle.dump(cnt, open("/root/data/sim2sim/len_500.pkl", "wb"))

# import numpy as np
# import os
# import sapien.core as sapien
# from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
# from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler

# p1 = np.random.randn(3)
# el1 = np.random.randn(3) * np.pi
# ps1 = sapien.Pose(p1, euler2quat(el1))
#
# p2 = np.random.randn(3)
# el2 = np.random.randn(3) * np.pi
# ps2 = sapien.Pose(p2, euler2quat(el2))
#
# p12 = p2 - p1
# q12 = qmult(ps2.q, qconjugate(ps1.q))
# print(p12, q12)
#
# q1_T_q2 = ps1.inv().transform(ps2)
# q2_T_q1 = ps2.transform(ps1.inv())
# print(q1_T_q2, q2_T_q1)
#
# mat1 = quat2mat(ps1.q)
# mat2 = quat2mat(ps2.q)
# mat1_inv = np.linalg.inv(mat1)
# mat12 = mat1_inv.dot(mat2)
# mat21 = mat2.dot(mat1_inv)
# q12 = mat2quat(mat12)
# q21 = mat2quat(mat21)
# print(q12, q21)
#
# print(ps1.inv(), ps1)
# T1 = np.r_[np.c_[mat1, p1], [[0, 0, 0, 1]]]
# T1_inv = np.linalg.inv(T1)
# print(T1)
# print(T1_inv)

# import sys
#
# sys.path.append("/root")
#
# from diffusion_policy.policy import DiffusionPolicy
import os
import pickle


success_list = []

data_root = "/root/data/cano_drawer_0915"

for seed in range(5000):
    seed_path = os.path.join(data_root, f"seed_{seed}")
    num_steps = len(os.listdir(seed_path)) // 2
    if num_steps >= 499:
        success_list.append((seed, "f", num_steps))
    else:
        success_list.append((seed, "s", num_steps))

pickle.dump(success_list, open(os.path.join(data_root, f"info.pkl"), "wb"))
