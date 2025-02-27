import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]
robot_xyzs = []
desired_xyzs = []
with open(fname, "rb") as f:
    try:
        while True:
            data = pickle.load(f)
            # print(data["elapsed_time"])
            print("control", data["control_state_time"], data["next_desired_pose"])
            print(
                "state",
                data["robot_state_time"],
                data["robot_xyz"],
                data["robot_joints"],
            )
            robot_xyzs.append(data["robot_xyz"])
            desired_xyzs.append(data["next_desired_pose"][:3, 3])
            # ax[0].cla()
            # ax[1].cla()
            # ax[0].imshow(data["rgb_head"])
            # ax[1].imshow(data["rgb_wrist"])
            # plt.pause(0.001)
            # video_writer.append_data(np.concatenate([data["rgb_head"], data["rgb_wrist"]], axis=1))
    except EOFError:
        pass
# video_writer.close()
robot_xyzs = np.array(robot_xyzs)
desired_xyzs = np.array(desired_xyzs)
fig, ax = plt.subplots(1, 3)
for i in range(3):
    ax[i].plot(robot_xyzs[:, i])
    ax[i].plot(desired_xyzs[:, i])
plt.savefig("debug.png")
