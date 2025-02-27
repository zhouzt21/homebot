## Dataset format for behavior cloning
Each trajectory is saved as a `[name].pkl` file containing a sequence of steps and a metadata file `[name]-meta.txt` containing the language label.
Each step is a `dict` formatted as follow:
- `rgb_head`: (H, W, C) image in the range `[0, 255]`.
- `rgb_wrist`: (H, W, C) image in the range `[0, 255]`.
- `robot_xyz`: (3,), the end effector position in the initial frame of the mobile base.
- `robot_rpy`: (3,), the end effector rotation in the initial frame of the mobile base.
- `robot_joints`: (7,), the joint positions of the arm.
- `gripper_width`: `float` in the range of `[0, 0.85]`.
