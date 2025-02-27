import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing import Any
from torch.utils.data import Dataset
from typing import Dict, List, Union, Optional
from homebot_sapien.utils.math import (
    wrap_to_pi,
    euler2mat,
    mat2euler,
    get_pose_from_rot_pos,
)


ALL_KEYS = ["rgb_head", "lang", "pose", "joint", "gripper_width"]
ALL_ROBOT_STATE_KEYS = ["pose", "joint", "gripper_width"]
OBS_NORMALIZE_PARAMS = {
    # use 6d representation for rotation, relative
    "pose": {
        "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "scale": np.array([0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    },
    "gripper_width": {"mean": np.array([0.425]), "scale": np.array([0.425])},
    "joint": {
        "mean": (
            np.array(
                [
                    -3.110177,
                    -2.180000,
                    -3.110177,
                    -0.110000,
                    -3.110177,
                    -1.750000,
                    -3.110177,
                ]
            )
            + np.array(
                [3.110177, 2.180000, 3.110177, 3.110177, 3.110177, 3.110177, 3.110177]
            )
        )
        / 2,
        "scale": (
            -np.array(
                [
                    -3.110177,
                    -2.180000,
                    -3.110177,
                    -0.110000,
                    -3.110177,
                    -1.750000,
                    -3.110177,
                ]
            )
            + np.array(
                [3.110177, 2.180000, 3.110177, 3.110177, 3.110177, 3.110177, 3.110177]
            )
        )
        / 2,
    },
}


class BCDataset(Dataset):
    def __init__(
        self,
        folder_name,
        n_images=1,
        file_sorted=False,
        robot_state_keys=(),
        # gripper_action_mode="delta_conti",
        image_wrist_or_head="head",
        obs_normalize_params=OBS_NORMALIZE_PARAMS,
        action_keys=("gripper", "pose"),
        # gripper_action_scale=1.0,
        action_relative="tool",
        action_lookahead=1,
        head_img_shift_range=(0, 0),
    ) -> None:
        super().__init__()
        # Each step is saved as a pickle file
        self.folder_name = folder_name
        self.file_name_list = os.listdir(folder_name)
        if file_sorted:

            def get_sort_key(s):
                prefix = s[:-4].split("step")[0]
                suffix = "%03d" % int(s[:-4].split("step")[1])
                key = prefix + suffix
                return key

            self.file_name_list = sorted(list(self.file_name_list), key=get_sort_key)
        self.n_images = n_images
        for key in robot_state_keys:
            assert key in ALL_ROBOT_STATE_KEYS
        self.robot_state_keys = robot_state_keys
        # self.gripper_action_mode = gripper_action_mode
        if len(self.robot_state_keys):
            self.robot_state_mean = np.concatenate(
                [obs_normalize_params[key]["mean"] for key in self.robot_state_keys]
            )
            self.robot_state_scale = np.concatenate(
                [obs_normalize_params[key]["scale"] for key in self.robot_state_keys]
            )
        else:
            self.robot_state_mean = np.zeros(0)
            self.robot_state_scale = np.ones(0)
        self.action_keys = action_keys
        # self.gripper_action_scale = gripper_action_scale
        self.action_relative = action_relative
        self.action_lookahead = action_lookahead
        self.image_wrist_or_head = image_wrist_or_head
        self.head_img_shift_range = head_img_shift_range
        self.p_scale = 0.05
        self.rot_scale = 0.2

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index: Any) -> Any:
        step_in_traj = int(self.file_name_list[index][:-4].split("step")[-1])
        prefix = self.file_name_list[index][:-4].split("step")[0]
        wrist_image_history = []
        head_image_history = []
        result_dict = {}
        for step in range(max(step_in_traj - self.n_images + 1, 0), step_in_traj + 1):
            file_name = prefix + "step%d.pkl" % step
            with open(os.path.join(self.folder_name, file_name), "rb") as f:
                data_dict = pickle.load(f)
            assert all([k in data_dict.keys() for k in ALL_KEYS])
            # TODO: Apply Random crop and shift to head images
            shift = np.random.randint(
                self.head_img_shift_range[0],
                self.head_img_shift_range[1] + 1,
                size=(2,),
            )
            head_image_history.append(
                self.random_shift_image(data_dict["rgb_head"], shift)
            )
            wrist_image_history.append(data_dict["rgb_wrist"])
            # if self.image_wrist_or_head == "head":
            #     image_history.append(data_dict["rgb_head"])
            # elif self.image_wrist_or_head == "wrist":
            #     image_history.append(data_dict["rgb_wrist"])
            # elif self.image_wrist_or_head == "both":
            #     image_history.append(data_dict["rgb_wrist"])
            #     image_history.append(data_dict["rgb_head"])
            if step == step_in_traj:
                result_dict = data_dict
                result_dict["lang"] = data_dict["lang"]
                if len(self.robot_state_keys):
                    robot_state = (
                        np.concatenate(
                            [data_dict[key] for key in self.robot_state_keys]
                        )
                        - self.robot_state_mean
                    ) / self.robot_state_scale
                else:
                    robot_state = np.zeros(0)
                result_dict["robot_state"] = robot_state
                next_file_name = prefix + "step%d.pkl" % (step + self.action_lookahead)
                if not os.path.exists(os.path.join(self.folder_name, next_file_name)):
                    is_terminate = True
                    next_data_dict = data_dict
                else:
                    is_terminate = False
                    with open(
                        os.path.join(self.folder_name, next_file_name), "rb"
                    ) as f:
                        next_data_dict = pickle.load(f)
                action_gripper = self.parse_gripper_action(data_dict, next_data_dict)
                action = np.zeros((0,))
                for key in self.action_keys:
                    if key == "is_terminate":
                        action = np.concatenate([action, np.array([is_terminate])])
                    elif key == "gripper":
                        action = np.concatenate([action, action_gripper])
                    elif key == "pose":
                        pose_action = self.parse_pose_action(data_dict, next_data_dict)
                        action = np.concatenate([action, pose_action])
                    elif key == "joint":
                        action = np.concatenate(
                            [action, next_data_dict["joint"] - data_dict["joint"]]
                        )
                    else:
                        raise NotImplementedError
                result_dict["action"] = action

        if step_in_traj - self.n_images + 1 < 0:
            wrist_image_history = np.concatenate(
                [
                    np.tile(
                        np.expand_dims(wrist_image_history[0], axis=0),
                        (-step_in_traj + self.n_images - 1, 1, 1, 1),
                    ),
                    np.array(wrist_image_history),
                ],
                axis=0,
            )
            head_image_history = np.concatenate(
                [
                    np.tile(
                        np.expand_dims(head_image_history[0], axis=0),
                        (-step_in_traj + self.n_images - 1, 1, 1, 1),
                    ),
                    np.array(head_image_history),
                ],
                axis=0,
            )
        else:
            wrist_image_history = np.array(wrist_image_history)
            head_image_history = np.array(head_image_history)
        if self.image_wrist_or_head == "wrist":
            result_dict["rgb"] = wrist_image_history
        elif self.image_wrist_or_head == "head":
            result_dict["rgb"] = head_image_history
        elif self.image_wrist_or_head == "both":
            result_dict["rgb"] = np.concatenate(
                [wrist_image_history, head_image_history], axis=0
            )
        else:
            raise NotImplementedError
        return result_dict
        # with open(
        #     os.path.join(self.folder_name, self.file_name_list[index]), "rb"
        # ) as f:
        #     data_dict: Dict = pickle.load(f)
        # assert all([k in data_dict.keys() for k in ALL_KEYS])
        # return data_dict

    def parse_gripper_action(self, data_dict, next_data_dict):
        # if self.gripper_action_mode == "delta_conti":
        #     action_gripper = (
        #         next_data_dict["gripper_width"] - data_dict["gripper_width"]
        #     )
        # elif self.gripper_action_mode == "3mode":
        #     if next_data_dict["gripper_width"] - data_dict["gripper_width"] > 0.05:
        #         action_gripper = np.array([self.gripper_action_scale])
        #     elif next_data_dict["gripper_width"] - data_dict["gripper_width"] < -0.05:
        #         action_gripper = np.array([-self.gripper_action_scale])
        #     else:
        #         action_gripper = np.array([0.0])
        # elif self.gripper_action_mode == "abs_conti":
        #     action_gripper = next_data_dict["gripper_width"] * self.gripper_action_scale
        # else:
        #     raise NotImplementedError
        # Same as simulation
        action_gripper = (0.85 - next_data_dict["gripper_width"]) / 0.85 * 2 - 1
        return action_gripper

    def parse_pose_action(self, data_dict, next_data_dict):
        pose_action = np.empty((6,))
        initbase_T_eef = get_pose_from_rot_pos(
            euler2mat(data_dict["pose"][3:]), data_dict["pose"][:3]
        )
        initbase_T_nexteef = get_pose_from_rot_pos(
            euler2mat(next_data_dict["pose"][3:]),
            next_data_dict["pose"][:3],
        )
        if self.action_relative == "tool":
            # Same as simulation
            cureef_T_nexteef = np.linalg.inv(initbase_T_eef) @ initbase_T_nexteef
            # translation
            pose_action[0:3] = np.clip(
                cureef_T_nexteef[:3, 3] / self.p_scale, -1.0, 1.0
            )
            # rotation
            pose_action[3:6] = np.clip(
                wrap_to_pi(mat2euler(cureef_T_nexteef[:3, :3])) / self.rot_scale,
                -1.0,
                1.0,
            )
        else:
            # Same as simulation
            pose_action[0:3] = np.clip(
                (next_data_dict["pose"][:3] - data_dict["pose"][:3]) / self.p_scale,
                -1.0,
                1.0,
            )
            pose_action[3:6] = np.clip(
                wrap_to_pi(next_data_dict["pose"][3:] - data_dict["pose"][3:])
                / self.rot_scale,
                -1.0,
                1.0,
            )
        return pose_action

    def random_shift_image(self, image: np.ndarray, shift: np.array):
        assert len(shift) == 2
        output = np.zeros_like(image)
        src_range = [
            np.clip(
                np.array([shift[0], image.shape[1] + shift[0]]), 0, image.shape[1]
            ).astype(np.int32),
            np.clip(
                np.array([shift[1], image.shape[2] + shift[1]]), 0, image.shape[2]
            ).astype(np.int32),
        ]
        tgt_range = [
            np.clip(
                np.array([-shift[0], image.shape[1] - shift[0]]), 0, image.shape[1]
            ).astype(np.int32),
            np.clip(
                np.array([-shift[1], image.shape[2] - shift[1]]), 0, image.shape[2]
            ).astype(np.int32),
        ]
        output[
            :, tgt_range[0][0] : tgt_range[0][1], tgt_range[1][0] : tgt_range[1][1]
        ] = image[
            :, src_range[0][0] : src_range[0][1], src_range[1][0] : src_range[1][1]
        ]
        return output


class EpisodeDataset(Dataset):
    def __init__(
        self,
        folder_name: Union[str, List[str]],
        chunk_size,
        data_config: dict,
        estimate_stats: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(folder_name, str):
            folder_name = [folder_name]
        self.folder_name = folder_name
        self.chunk_size = chunk_size
        self.data_config = data_config
        self.traj_file_list = []
        for _folder_name in self.folder_name:
            if os.path.exists(_folder_name):
                flist = os.listdir(_folder_name)
                self.traj_file_list.extend(
                    [os.path.join(_folder_name, fname) for fname in flist]
                )
        # self.traj_file_list = os.listdir(folder_name)
        self.traj_step = []
        self._item_length_bytes = None
        joint_lower = np.inf * np.ones(7)
        joint_upper = -np.inf * np.ones(7)
        for file_name in self.traj_file_list:
            step_count = 0
            with open(file_name, "rb") as f:
                try:
                    while True:
                        _data = pickle.load(f)
                        joint_lower = np.minimum(_data["robot_joints"], joint_lower)
                        joint_upper = np.maximum(_data["robot_joints"], joint_upper)
                        if self._item_length_bytes is None:
                            self._item_length_bytes = f.tell()
                        step_count += 1
                except EOFError:
                    pass
            self.traj_step.append(step_count)
        self.traj_step = np.array(self.traj_step)
        self.traj_step_cumsum = np.cumsum(self.traj_step)

        obs_normalize_params = copy.deepcopy(OBS_NORMALIZE_PARAMS)
        if estimate_stats:
            stats = self.compute_normalize_stats()
            for key in stats:
                obs_normalize_params[key] = stats[key]
            print(obs_normalize_params)
        self.update_obs_normalize_params(obs_normalize_params)

    def update_obs_normalize_params(self, obs_normalize_params: Optional[dict]):
        if obs_normalize_params is None:
            obs_normalize_params = OBS_NORMALIZE_PARAMS
        self.OBS_NORMALIZE_PARAMS = copy.deepcopy(obs_normalize_params)
        self.joint_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["joint", "gripper_width"]
            ]
        )
        self.joint_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["joint", "gripper_width"]
            ]
        )
        self.pose_gripper_mean = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["mean"]
                for key in ["pose", "gripper_width"]
            ]
        )
        self.pose_gripper_scale = np.concatenate(
            [
                self.OBS_NORMALIZE_PARAMS[key]["scale"]
                for key in ["pose", "gripper_width"]
            ]
        )

    def __len__(self):
        return self.traj_step_cumsum[-1]

    def __getitem__(self, index: int) -> Any:
        result = self.get_unnormalized_item(index)
        proprio_state = result["robot_state"]
        is_pad = result["is_pad"]
        action_chunk = result["action"]
        if "joint" in self.data_config["robot_state_keys"]:
            proprio_state = (
                proprio_state - self.joint_gripper_mean
            ) / self.joint_gripper_scale
            action_chunk[~is_pad] = (
                action_chunk[~is_pad] - np.expand_dims(self.joint_gripper_mean, axis=0)
            ) / np.expand_dims(self.joint_gripper_scale, axis=0)
        elif "pose" in self.data_config["robot_state_keys"]:
            proprio_state = (
                proprio_state - self.pose_gripper_mean
            ) / self.pose_gripper_scale
            action_chunk[~is_pad] = (
                action_chunk[~is_pad] - np.expand_dims(self.pose_gripper_mean, axis=0)
            ) / np.expand_dims(self.pose_gripper_scale, axis=0)
        result["robot_state"] = proprio_state
        result["action"] = action_chunk
        return result

    def get_unnormalized_item(self, index: int):
        result_dict = {}
        result_dict["lang"] = " "
        traj_idx, start_ts = self._locate(index)
        # image: start_ts
        # joint: [start_ts, start_ts + chunk_size]
        end_ts = min(self.traj_step[traj_idx], start_ts + self.chunk_size + 1)
        is_pad = np.zeros((self.chunk_size,), dtype=bool)
        if end_ts < start_ts + self.chunk_size + 1:
            is_pad[-(start_ts + self.chunk_size + 1 - end_ts) :] = True
        result_dict["is_pad"] = is_pad
        if "joint" in self.data_config["action_keys"]:
            action_chunk = np.zeros((self.chunk_size, 8), dtype=np.float32)
            joint_position_chunk = []
            gripper_width_chunk = []
        elif "pose" in self.data_config["action_keys"]:
            action_chunk = np.zeros((self.chunk_size, 10), dtype=np.float32)
            pose_chunk = []
            gripper_width_chunk = []
        else:
            raise NotImplementedError
        if "joint" in self.data_config["robot_state_keys"]:
            proprio_state = np.zeros((8,), dtype=np.float32)
        elif "pose" in self.data_config["robot_state_keys"]:
            proprio_state = np.zeros((10,), dtype=np.float32)
        else:
            raise NotImplementedError
        file_path = self.traj_file_list[traj_idx]
        result_dict["rgb"] = []
        history_obs_ts = np.arange(
            start_ts - (self.data_config["n_images"] - 1), start_ts + 1, 1
        )  # the history is stacked at control frequency
        # handle out of range
        history_obs_ts = np.maximum(history_obs_ts, 0)
        # Get rgb images
        with open(file_path, "rb") as f:
            for _t in history_obs_ts:
                f.seek(self._item_length_bytes * _t)
                data = pickle.load(f)
                result_dict["rgb"].append(
                    np.stack(
                        [
                            np.transpose(data["rgb_wrist"], (2, 0, 1)),
                            np.transpose(data["rgb_head"], (2, 0, 1)),
                        ],
                        axis=0,
                    )
                )
        result_dict["rgb"] = np.concatenate(result_dict["rgb"], axis=0)
        # Get other observations and future poses
        with open(file_path, "rb") as f:
            f.seek(self._item_length_bytes * start_ts)
            for step_idx in range(start_ts, end_ts):
                data = pickle.load(f)
                if step_idx == start_ts:
                    # result_dict["rgb"] = np.stack(
                    #     [
                    #         np.transpose(data["rgb_wrist"], (2, 0, 1)),
                    #         np.transpose(data["rgb_head"], (2, 0, 1)),
                    #     ],
                    #     axis=0,
                    # )
                    if "joint" in self.data_config["robot_state_keys"]:
                        proprio_state[:] = np.concatenate(
                            [data["robot_joints"], np.array([data["gripper_width"]])]
                        )
                        if "next_desired_joints" in data:
                            joint_position_chunk.append(data["next_desired_joints"])
                        else:
                            print("Warning, no next_desired_joints recorded")
                        if "next_desired_gripper_width" in data:
                            gripper_width_chunk.append(
                                data["next_desired_gripper_width"]
                            )
                        else:
                            print("Warning, no next_desired_gripper_width recorded")
                    elif "pose" in self.data_config["robot_state_keys"]:
                        pose_at_obs = get_pose_from_rot_pos(
                            euler2mat(data["robot_rpy"]), data["robot_xyz"]
                        )
                        pose_rot = euler2mat(data["robot_rpy"])[:, :2].reshape(-1)
                        proprio_state[:] = np.concatenate(
                            [
                                data["robot_xyz"],
                                pose_rot,
                                np.array([data["gripper_width"]]),
                            ]
                        )
                        if "next_desired_pose" in data:
                            pose_chunk.append(data["next_desired_pose"])
                elif step_idx > start_ts:
                    if "joint" in self.data_config["action_keys"]:
                        if "next_desired_joints" in data:
                            joint_position_chunk.append(data["next_desired_joints"])
                        else:
                            print("Warning, no next_desired_joints recorded")
                            joint_position_chunk.append(data["robot_joints"])
                        if "next_desired_gripper_width" in data:
                            gripper_width_chunk.append(
                                data["next_desired_gripper_width"]
                            )
                        else:
                            print("Warning, no next_desired_gripper_width recorded")
                            gripper_width_chunk.append(data["gripper_width"])
                        # action_chunk[step_idx - (start_ts + 1)] = np.concatenate(
                        #     [data["robot_joints"], np.array([data["gripper_width"]])]
                        # )
                    elif "pose" in self.data_config["action_keys"]:
                        # If the desired pose is recorded, we should use it. Otherwise, fall back to using the future achieved pose
                        if "next_desired_pose" in data:
                            pose_chunk.append(data["next_desired_pose"])
                        else:
                            pose_chunk.append(
                                get_pose_from_rot_pos(
                                    euler2mat(data["robot_rpy"]), data["robot_xyz"]
                                )
                            )
                        gripper_width_chunk.append(np.array([data["gripper_width"]]))
                        pose_rot = euler2mat(data["robot_rpy"])[:, :2].reshape(-1)
                        # Should we remove?
                        # action_chunk[step_idx - (start_ts + 1)] = np.concatenate(
                        #     [
                        #         data["robot_xyz"],
                        #         pose_rot,
                        #         np.array([data["gripper_width"]]),
                        #     ]
                        # )
        # normalize state and action
        if "joint" in self.data_config["robot_state_keys"]:
            # proprio_state = (
            #     proprio_state - self.joint_gripper_mean
            # ) / self.joint_gripper_scale
            action_chunk[: (end_ts - start_ts - 1)] = np.concatenate(
                [
                    np.array(joint_position_chunk)[: (end_ts - start_ts - 1)],
                    np.expand_dims(
                        np.array(gripper_width_chunk)[: (end_ts - start_ts - 1)],
                        axis=-1,
                    ),
                ],
                axis=-1,
            )
            # action_chunk[: (end_ts - start_ts - 1)] = (
            #     action_chunk[: (end_ts - start_ts - 1)]
            #     - np.expand_dims(self.joint_gripper_mean, axis=0)
            # ) / np.expand_dims(self.joint_gripper_scale, axis=0)
        elif "pose" in self.data_config["robot_state_keys"]:
            # make relative
            _pose_relative = np.eye(4)
            proprio_state[:9] = np.concatenate(
                [_pose_relative[:3, 3], _pose_relative[:3, :2].reshape(-1)]
            )
            for i in range(end_ts - start_ts - 1):
                _pose_relative = np.linalg.inv(pose_at_obs) @ pose_chunk[i]
                action_chunk[i] = np.concatenate(
                    [
                        _pose_relative[:3, 3],
                        _pose_relative[:3, :2].reshape(-1),
                        gripper_width_chunk[i],
                    ]
                )
            # proprio_state = (
            #     proprio_state - self.pose_gripper_mean
            # ) / self.pose_gripper_scale
            # action_chunk[: (end_ts - start_ts - 1)] = (
            #     action_chunk[: (end_ts - start_ts - 1)]
            #     - np.expand_dims(self.pose_gripper_mean, axis=0)
            # ) / np.expand_dims(self.pose_gripper_scale, axis=0)
        result_dict["robot_state"] = proprio_state
        result_dict["action"] = action_chunk
        return result_dict

    def _locate(self, index: int):
        assert index < len(self)
        traj_idx = np.where(self.traj_step_cumsum > index)[0][0]
        till_last_cumstep = self.traj_step_cumsum[traj_idx - 1] if traj_idx > 0 else 0
        start_ts = index - till_last_cumstep
        return traj_idx, start_ts

    def compute_normalize_stats(self, scale_eps=0.05):
        # min and max scale
        joint_min, joint_max = None, None
        gripper_width_min, gripper_width_max = None, None
        pose_min, pose_max = None, None

        def safe_minimum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.minimum(a, b)

        def safe_maximum(a: np.ndarray, b: np.ndarray):
            if a is None:
                return b
            if b is None:
                return a
            return np.maximum(a, b)

        def safe_min(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.min(a, axis=axis)

        def safe_max(a: np.ndarray, axis: int):
            if a.shape[axis] == 0:
                return None
            return np.max(a, axis=axis)

        for i in range(len(self)):
            item_dict = self.get_unnormalized_item(i)
            if "joint" in self.data_config["robot_state_keys"]:
                joint_position = item_dict["robot_state"][:7]
                action_joint_positions = item_dict["action"][~item_dict["is_pad"]][
                    :, :7
                ]
                gripper_width = item_dict["robot_state"][7:8]
                action_gripper_width = item_dict["action"][~item_dict["is_pad"]][:, 7:8]
                joint_min = safe_minimum(
                    safe_minimum(joint_min, joint_position),
                    safe_min(action_joint_positions, axis=0),
                )
                joint_max = safe_maximum(
                    safe_maximum(joint_max, joint_position),
                    safe_max(action_joint_positions, axis=0),
                )
                gripper_width_min = safe_minimum(
                    safe_minimum(gripper_width_min, gripper_width),
                    safe_min(action_gripper_width, axis=0),
                )
                gripper_width_max = safe_maximum(
                    safe_maximum(gripper_width_max, gripper_width),
                    safe_max(action_gripper_width, axis=0),
                )

            elif "pose" in self.data_config["robot_state_keys"]:
                pose = item_dict["robot_state"][:9]
                action_pose = item_dict["action"][~item_dict["is_pad"]][:, :9]
                gripper_width = item_dict["robot_state"][9:10]
                action_gripper_width = item_dict["action"][~item_dict["is_pad"]][
                    :, 9:10
                ]
                pose_min = safe_minimum(
                    safe_minimum(pose_min, pose), safe_min(action_pose, axis=0)
                )
                pose_max = safe_maximum(
                    safe_maximum(pose_max, pose), safe_max(action_pose, axis=0)
                )
                gripper_width_min = safe_minimum(
                    safe_minimum(gripper_width_min, gripper_width),
                    safe_min(action_gripper_width, axis=0),
                )
                gripper_width_max = safe_maximum(
                    safe_maximum(gripper_width_max, gripper_width),
                    safe_max(action_gripper_width, axis=0),
                )

        params = {}
        if pose_min is not None:
            params["pose"] = {
                "mean": (pose_min + pose_max) / 2,
                "scale": np.maximum((pose_max - pose_min) / 2, scale_eps),
            }
        if joint_min is not None:
            params["joint"] = {
                "mean": (joint_min + joint_max) / 2,
                "scale": np.maximum((joint_max - joint_min) / 2, scale_eps),
            }
        params["gripper_width"] = {
            "mean": (gripper_width_min + gripper_width_max) / 2,
            "scale": np.maximum((gripper_width_max - gripper_width_min) / 2, scale_eps),
        }
        return params


def step_collate_fn(samples: List[Dict]):
    batch = {}
    for key in samples[0].keys():
        if key != "lang":
            batched_array = np.array([sample[key] for sample in samples])
            batch[key] = batched_array
    batch["lang"] = [sample["lang"] for sample in samples]
    return batch


def export_video_from_demonstration(file_name, gif_name):
    import imageio

    all_rgb = []
    all_xyz = []
    all_width = []
    with open(file_name, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                rgb = data["rgb_head"].astype(np.uint8)
                robot_xyz = data["robot_xyz"]
                gripper_width = data["gripper_width"]
                if len(all_rgb) == 0 or np.linalg.norm(robot_xyz - all_xyz[-1]) > 1e-4:
                    all_rgb.append(rgb)
                    all_xyz.append(robot_xyz)
                    all_width.append(gripper_width)
            except EOFError:
                break
    trim_idx = len(all_width)
    # trim_idx = (
    #     np.where(
    #         [
    #             all_width[i + 2] < 0.7 and all_width[i] > 0.7
    #             for i in range(len(all_width) - 2)
    #         ]
    #     )[0][-1]
    #     + 2
    # )
    imageio.mimwrite(gif_name, all_rgb[:trim_idx], format=".gif", fps=20)


def play_demonstration(
    file_name,
    # relative_action=False,
    # gripper_action="delta_conti",
    # pose_or_joint="pose",
    out_folder="dataset",
):
    """
    file_name: pkl file path
    """
    meta_file_name = file_name[:-4] + "-meta.txt"
    if not os.path.exists(meta_file_name):
        lang = ""
    else:
        with open(meta_file_name, "r") as f:
            lang = f.read()
        print(lang)
    all_xyz, all_rpy, all_joints, all_width = [], [], [], []
    all_rgb_wrist, all_rgb_head = [], []
    all_elapsed_time = []
    # fig, ax = plt.subplots(1, 1)
    # Do not add the steps where the robot does not move
    # Manually mark the key frame idx
    with open(file_name, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                if "rgb_head" in data:
                    rgb_head = data["rgb_head"].astype(np.uint8)
                    rgb_wrist = data["rgb_wrist"].astype(np.uint8)
                else:
                    print(
                        "Warning: this trajectory does not have head camera image. Skip it"
                    )
                    return
                if "control_mode" in data and data["control_mode"] == "robot":
                    continue
                robot_xyz = data["robot_xyz"]
                robot_rpy = data["robot_rpy"]
                robot_joints = data["robot_joints"]
                gripper_width = data["gripper_width"]
                if len(all_xyz) == 0 or np.linalg.norm(robot_xyz - all_xyz[-1]) > 1e-4:
                    all_rgb_wrist.append(rgb_wrist)
                    all_rgb_head.append(rgb_head)
                    all_xyz.append(robot_xyz)
                    all_rpy.append(robot_rpy)
                    all_joints.append(robot_joints)
                    all_width.append(gripper_width)
                    # all_elapsed_time.append(data["elapsed_time"])
            except EOFError:
                break
            # ax.cla()
            # ax.imshow(rgb)
            # plt.pause(0.1)
    all_xyz = np.array(all_xyz)
    all_rpy = np.array(all_rpy)
    # fig, ax = plt.subplots(2, 3)
    # for i in range(3):
    #     ax[0][i].plot(all_xyz[:, i])
    #     ax[1][i].plot(all_rpy[:, i])
    # plt.show()
    all_width = np.array(all_width)
    # Do not trim for general cases
    trim_idx = len(all_width)
    # if len(all_width) == 0:
    #     return
    # try:
    #     trim_idx = (
    #         np.where(
    #             [
    #                 all_width[i + 2] > 0.8 and all_width[i] < 0.8
    #                 for i in range(len(all_width) - 2)
    #             ]
    #         )[0][-1]
    #         + 2
    #     )
    # except:
    #     print("all_width", all_width)
    #     trim_idx = int(input("trim idx should be:").strip())
    print("trim idx", trim_idx)
    # ax.plot(all_width)
    # plt.show()

    # keyframe_file_name = file_name[:-4] + "-keyframe.txt"
    # if os.path.exists(keyframe_file_name):
    #     with open(keyframe_file_name, "r") as f:
    #         res = f.read()
    # else:
    #     changing_frames = np.where(
    #         [abs(all_width[i + 1] - all_width[i]) > 0.1 for i in range(trim_idx)]
    #     )[0]
    #     print(changing_frames, all_width[changing_frames])
    #     res = input("key frames are: seprate by space")
    #     with open(keyframe_file_name, "w") as f:
    #         f.write(res)
    # key_frames = res.strip().split(" ")
    # try:
    #     key_frames = [int(item) for item in key_frames] + [trim_idx]
    # except:
    #     print(keyframe_file_name)
    #     print("res", res)
    #     raise RuntimeError

    if out_folder and not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    # fig, ax = plt.subplots(1, 1)
    # data_frames = [0]
    for i in range(trim_idx):
        # for i in data_frames:
        # if i >= trim_idx:
        #     break
        # Dump state (observation) and action
        rgb_wrist = np.transpose(all_rgb_wrist[i], (2, 0, 1))
        rgb_head = np.transpose(all_rgb_head[i], (2, 0, 1))
        # ax.cla()
        # ax.imshow(all_rgb_head[i])
        # print("elapsed time", all_elapsed_time[i])
        # plt.pause(0.05)
        new_file_name = os.path.basename(file_name)[:-4] + "-step%d" % i + ".pkl"
        # if i >= key_frames[0]:
        #     key_frames.pop(0)
        # next_key_frame_idx = key_frames[0]

        with open(os.path.join(out_folder, new_file_name), "wb") as f:
            pickle.dump(
                {
                    "rgb_wrist": rgb_wrist,
                    "rgb_head": rgb_head,
                    # "robot_state": robot_state,
                    "pose": np.concatenate([all_xyz[i], all_rpy[i]]),
                    "joint": all_joints[i],
                    "gripper_width": np.array([all_width[i]]),
                    "lang": lang,
                    # "next_key_frame_pose": np.concatenate(
                    #     [all_xyz[next_key_frame_idx], all_rpy[next_key_frame_idx]]
                    # ),
                    # "next_key_frame_joint": all_joints[next_key_frame_idx],
                    # "next_key_frame_gripper_width": np.array(
                    #     [all_width[next_key_frame_idx]]
                    # ),
                    # "action": action,
                },
                f,
            )
    # plt.close(fig)
    # print("max action", max_action)


def preprocess_demonstration(file_name, out_folder="dataset"):
    lang_instruction = file_name.split("-")[0]
    lang_instruction = lang_instruction.replace("_", " ")
    mobile_T_rbase = get_pose_from_rot_pos(
        euler2mat(np.array([0.0, 0.0, np.pi / 2])), np.array([0.0, 0.0, 0.62])
    )
    traj = []
    last_robot_eef = None
    last_gripper_width = None
    with open(file_name, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
                rgb = np.transpose(data["rgb"], (2, 0, 1)).astype(np.uint8)
                robot_state = np.concatenate(
                    [data["robot_joints"], [data["gripper_width"]]]
                )
                traj.append(
                    {"rgb": rgb, "robot_state": robot_state, "lang": lang_instruction}
                )
                action = np.zeros(8)
                # use mobile base state to compute eef pose in world
                # the agent does not know world coordinate, maybe modify to relative position in observation tool coordinate
                mobile_base_xytheta = data["agv_xytheta"]
                O_T_mobile = get_pose_from_rot_pos(
                    euler2mat(np.array([0.0, 0.0, mobile_base_xytheta[2]])),
                    np.array([mobile_base_xytheta[0], mobile_base_xytheta[1], 0.0]),
                )
                rbase_T_eef = get_pose_from_rot_pos(
                    euler2mat(data["robot_rpy"]), data["robot_xyz"]
                )
                O_T_eef = O_T_mobile @ mobile_T_rbase @ rbase_T_eef
                new_robot_eef = O_T_eef
                new_gripper_width = data["gripper_width"]
                if len(traj) > 1:
                    if np.abs(new_gripper_width - last_gripper_width) > 0.1:
                        action[1] = 1.0
                    else:
                        lasteef_T_cureef = np.linalg.inv(last_robot_eef) @ O_T_eef
                        # clip and scale to [-1, 1]
                        action[2:5] = np.clip(lasteef_T_cureef[:3, 3] / 0.5, -1.0, 1.0)
                        action[5:8] = (
                            wrap_to_pi(mat2euler(lasteef_T_cureef[:3, :3])) / np.pi
                        )
                    traj[-2]["action"] = action
                    # print(traj[-2]["robot_state"], traj[-2]["lang"], traj[-2]["action"])
                last_robot_eef = new_robot_eef
                last_gripper_width = new_gripper_width
            except EOFError:
                break
    final_action = np.zeros(8)
    final_action[0] = 1.0
    traj[-1]["action"] = final_action
    # print(traj[-1]["robot_state"], traj[-1]["lang"], traj[-1]["action"])
    # Save each step as a file
    if out_folder and not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    for i in range(len(traj)):
        new_file_name = file_name[:-4] + "-step%d" % i + ".pkl"
        with open(os.path.join(out_folder, new_file_name), "wb") as f:
            pickle.dump(traj[i], f)
    return traj
