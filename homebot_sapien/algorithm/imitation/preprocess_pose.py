import numpy as np
import sapien.core as sapien
import os
import pickle
import sys
from homebot_sapien.utils.make_env import make_env
from homebot_sapien.utils.math import quat2euler

env = make_env("Opendoor-v0")
env.robot.set_root_pose(
    sapien.Pose(
        p=np.array([10.0, 0.0, -0.62]),
        q=np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)]),
    )
)
folder_name = sys.argv[1]
basename_list = os.listdir(folder_name)
for basename in basename_list:
    fname = os.path.join(folder_name, basename)
    new_fname = os.path.join(os.path.dirname(fname) + "fixed", os.path.basename(fname))
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname), exist_ok=True)
    new_f = open(new_fname, "ab")
    with open(fname, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                joint_position = data["robot_joints"]
                qpos = np.zeros_like(env.robot.get_qpos())
                qpos[env.base_arm_controller.arm_joint_indices] = joint_position
                env.base_arm_controller.pmodel.compute_forward_kinematics(qpos)
                ee_pose = env.base_arm_controller.pmodel.get_link_pose(
                    env.base_arm_controller.ee_link_idx
                )
                ee_T_tcp_pose = sapien.Pose(
                    p=np.array([0.0, 0.0, 0.172]),
                )
                robot_T_tcp_pose = (
                    sapien.Pose(
                        p=np.array([0.0, 0.0, 0.62]),
                        q=np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
                    )
                    .inv()
                    .transform(ee_pose)
                    .transform(ee_T_tcp_pose)
                )
                O_T_robot = sapien.Pose(
                    p=np.array([0.0, 0.0, 0.6113]),
                    q=np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
                )
                O_T_tcp = O_T_robot.transform(robot_T_tcp_pose)
                print(
                    "recomputed",
                    O_T_tcp.to_transformation_matrix(),
                    "command",
                    data["next_desired_pose"],
                )
                data["robot_xyz"] = O_T_tcp.p
                data["robot_rpy"] = quat2euler(O_T_tcp.q)
                pickle.dump(data, new_f)
            except EOFError:
                break
    new_f.close()
