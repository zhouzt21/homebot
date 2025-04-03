import cv2
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
from collections import OrderedDict
import sys 
sys.path.append("/home/zhouzhiting/Projects/homebot")
from homebot_sapien.env.base import BaseEnv, recover_action, get_pairwise_contact_impulse, get_pairwise_contacts

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from homebot_sapien.env.utils import apply_random_texture, check_intersect_2d, grasp_pose_process, check_intersect_2d_
from homebot_sapien.env.articulation.pick_and_place_articulation import (
    load_lab_wall,
    load_table_4,
    load_storage_box,
    build_actor_real,
    ASSET_DIR
)
# from homebot_sapien.env.drawer_articulation import (
#     load_table_4,
# )
from homebot_sapien.env.robot import load_robot_panda
from homebot_sapien.env.controller.whole_body_controller import ArmSimpleController
import json


# from Projects.homebot.config import PANDA_DATA
PANDA_DATA = "/home/zhouzhiting/Data/panda_data"

class PickAndPlaceEnv(BaseEnv):
    def __init__(
            self,
            use_gui: bool,
            device: str,
            mipmap_levels=1,
            obs_keys=tuple(),
            action_relative="tool",
            domain_randomize=True,
            canonical=True,
            allow_dir = []
    ):
        self.tcp_link_idx: int = None
        self.agv_link_idx: int = None
        self.door_handle_link_idx: int = None
        self.finger_link_idxs: List[int] = None
        self.observation_dict: dict = {}
        self.obs_keys = obs_keys
        self.action_relative = action_relative
        self.expert_phase = 0
        self.domain_randomize = domain_randomize
        self.canonical = canonical 
        self.allow_dir = allow_dir
        super().__init__(use_gui, device, mipmap_levels)

        cam_p = np.array([0.763579180, -0.03395012, 1.44071344])    # calibrate   0.793, -0.056, 1.505
        look_at_dir = np.array([-0.53301526,  0.01688062,  -0.84593722])  # rotate  -0.616, 0.044, -0.787
        right_dir = np.array([0.05021884, 0.99866954, -0.01171393])  #0.036, 0.999, 0.027
        self.create_camera(
            position=cam_p,
            look_at_dir=look_at_dir,
            right_dir=right_dir,
            name="third",
            resolution=(640, 480),
            fov=np.deg2rad(44),
        )


        self.standard_head_cam_pose = self.cameras["third"].get_pose()
        self.standard_head_cam_fovx = self.cameras["third"].fovx

        self.arm_controller = ArmSimpleController(self.robot)
        self.p_scale = 0.01
        self.rot_scale = 0.04
        self.gripper_scale = 0.007
        self.gripper_limit = 0.04

        joint_low = np.array(
            [
                -2.9671,
                -1.8326,
                -2.9671,
                -3.1416,
                -2.9671,
                -0.0873,
                -2.9671,
            ]
        )
        joint_high = np.array(
            [
                2.9671,
                1.8326,
                2.9671,
                0.0,
                2.9671,
                3.8223,
                2.9671
            ]
        )
        self.reset_joint_values = np.array(
            # [0., -0.785, 0., -2.356, 0., 1.571, 0.785]
            [0., -0.85, 0., -2.8, 0., 2.1, 0.785]
        )
        self.joint_scale = (joint_high - joint_low) / 2
        self.joint_mean = (joint_high + joint_low) / 2
        # Set spaces
        self.model_db = json.load(open(os.path.join(ASSET_DIR, "real_assets", "info_pick_v2.json"), "r"))  # v2

        self.reset(seed=0)
        self._update_observation()
        _obs_space_dict = {}
        for key in self.obs_keys:
            _obs_space_dict[key] = gym.spaces.Box(
                low=0 if "rgb" in key else -100,
                high=255 if "rgb" in key else 100,
                shape=self.observation_dict[key].shape,
                dtype=self.observation_dict[key].dtype,
            )
        self.observation_space = gym.spaces.Dict(_obs_space_dict)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def load_scene(self):
        self.load_static()

        self.robot, self.finger_link_idxs = load_robot_panda(self.scene)

        self.table_top_z = 0.76
        self.tray_cano_size = np.array([0.15, 0.12, 0.01])
        self.tray = load_storage_box(
            self.scene,
            box_size=self.tray_cano_size,
            thickness=0.003,
            root_position=np.array([0.4, -0.2, self.table_top_z]),
        )
        self.tray_info = dict(
            tray_size=self.tray_cano_size,
            tray_position=np.array([0.4, -0.2, self.table_top_z]),
            tray_yaw=0.
        )

    def load_static(self):
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.light0 = self.scene.add_directional_light(
            np.array([1, -1, -1]), np.array([1.0, 1.0, 1.0]), shadow=True
        )
        # self.scene.add_directional_light([1, 0, -1], [0.9, 0.8, 0.8], shadow=False)
        self.scene.add_directional_light([0, 1, 1], [0.9, 0.8, 0.8], shadow=False)

        physical_material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        render_material = self.renderer.create_material()
        render_material.set_base_color(np.array([0.1, 0.1, 0.1, 1.0]))
        self.room_ground = self.scene.add_ground(
            0.0, material=physical_material, render_material=render_material
        )
        ###
        self.table4 = load_table_4(self.scene)
        # self.table2 = load_table_2(self.scene, root_position=np.array([1., 2., 0]))
        # Add room walls
        self.room_wall1 = load_lab_wall(self.scene, [-1.0, 0.0], 10.0)
        # self.room_wall2 = load_lab_wall(self.scene, [0.0, -5.0], 10.0, np.pi / 2)

        self.walls = [
            self.room_wall1,  # self.room_wall2,  # self.room_wall3,
        ]
        self.tables = [
            self.table4,  # self.table2,
        ]

    def reload_objs(self, obj_list=None):
        if hasattr(self, "objs"):
            for obj_id, obj_dict in self.objs.items():
                self.scene.remove_actor(obj_dict["actor"])
            self.objs.clear()
        else:
            self.objs = dict()

        real_list = [("real", model_id) for model_id in self.model_db.keys()]
        obj_list = self.np_random.choice( real_list , 1, replace=False)
        
        for model_type, model_id in obj_list:
            obj_info = self.model_db[model_id]
            num_try = 0
            obj_invalid, init_p, init_angle, obj, init_scale = True, None, None, None, None
            while num_try < 10 and obj_invalid:
                rand_radius = self.np_random.uniform(0.03, 0.06)
                rand_ratio = self.np_random.uniform(0.9, 1.1)
                init_scale = np.array([
                    rand_radius / obj_info["bbox"]["max"][0],
                    rand_radius / obj_info["bbox"]["max"][1],
                    rand_radius * rand_ratio / (obj_info["bbox"]["max"][-1] * 2)
                ]) * obj_info["scales"][0]

                bbox_min_z = obj_info["bbox"]["min"][-1] * init_scale[-1]

                rand_p = self.np_random.uniform(-1, 1, size=(2,))
                init_p = np.array([0.4 + 0.15 * rand_p[0], 0.3 * rand_p[1], self.table_top_z - bbox_min_z + 5e-3])

                init_angle = self.np_random.uniform(-np.pi, np.pi)
                init_q = np.array(
                    [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]
                )

                dist = np.sqrt(((init_p[:2] - self.tray_info["tray_position"][:2]) ** 2).sum())
                if dist > self.tray_info["tray_size"][:2].max() + self.obj_infos[obj_id]["bbox"]["max"][0] * init_scale[0]:
                    obj_invalid = False

                num_try += 1
            # TODO

            assert model_type == "real", f"model_type {model_type} not supported"
            obj_allow_dir = obj_info["allow_dir"] 

            if not obj_invalid:
                obj = build_actor_real(
                    obj_id, self.scene, root_position=init_p,  root_rot=init_q,
                    density=obj_info.get("density", 1000),
                    scale=init_scale,
                    obj_allow_dir=obj_allow_dir
                )
                obj.set_damping(0.1, 0.1)

                self.objs.update({obj_id: dict(actor=obj, init_p=init_p, init_q = init_q, init_scale=init_scale)})

    def reset(self, seed: int = None, options: dict = None, obj_list: List = None):
        super().reset(seed, options)
        self.canonical_random, _ = seeding.np_random(seed)

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            # table
            table_rand = self.np_random.uniform(-1, 1, (2,))
            table_rand_size = np.array([0.1 * table_rand[0], 0.1 * table_rand[1], 0.]) + np.array([0.8, 1.0, 0.03])
            leg_pos_x = table_rand_size[0] / 2 - 0.1
            leg_pos_y = table_rand_size[1] / 2 - 0.1
            table_position = np.array([0.23, 0, 0])
            if hasattr(self, "table4"):
                self.scene.remove_actor(self.table4)
            self.table4 = load_table_4(
                self.scene,
                surface_size=table_rand_size,
                leg_pos_x=leg_pos_x,
                leg_pos_y=leg_pos_y,
                root_position=table_position,
            )
            self.tables = [self.table4]

            # walls
            wall_rand = self.np_random.uniform(0, 1, (2,))
            wall1_rand = table_position[0] - table_rand_size[0] / 2 - 0.3 - wall_rand[0] * 0.5
            # print(wall1_rand)
            # wall2_rand = -1.5 - wall_rand[1] * 3
            if hasattr(self, "room_wall1"):
                self.scene.remove_actor(self.room_wall1)
            # if hasattr(self, "room_wall2"):
            #     self.scene.remove_actor(self.room_wall2)
            self.room_wall1 = load_lab_wall(self.scene, [wall1_rand, 0.0], 10.0)
            # self.room_wall2 = load_lab_wall(self.scene, [0.0, wall2_rand], 10.0, np.pi / 2)
            self.walls = [
                self.room_wall1,  # self.room_wall2,  # self.room_wall3,
            ]

            # tray
            tray_size_rand = self.np_random.uniform(-1, 1, (2,))
            tray_pose_rand = self.np_random.uniform(-1, 1, (3,))
            tray_rand_size = self.tray_cano_size + np.array([0.03 * tray_size_rand[0], 0.03 * tray_pose_rand[1], 0])
            tray_rand_position = np.array([0.4 + 0.1 * tray_pose_rand[0], 0.2 * tray_pose_rand[1], self.table_top_z])
            tray_rand_yaw = np.pi / 2 * tray_pose_rand[-1]
            if hasattr(self, "tray"):
                self.scene.remove_actor(self.tray)
            self.tray = load_storage_box(
                self.scene,
                box_size=tray_rand_size,
                thickness=0.003,
                # color=np.array([0.6, 0.0, 0.0]),
                color=np.array([0.6, 0.3, 0.0]),
                root_position=tray_rand_position,
                root_angle=tray_rand_yaw
            )
            self.tray_info = dict(
                tray_size=tray_rand_size,
                tray_position=tray_rand_position,
                tray_yaw=tray_rand_yaw
            )

            if not self.canonical:
                self.scene.set_ambient_light(np.tile(self.canonical_random.uniform(0, 1, (1,)), 3))
                self.scene.remove_light(self.light0)
                self.light0 = self.scene.add_directional_light(
                    self.canonical_random.uniform(-1, 1, (3,)),
                    np.array([1.0, 1.0, 1.0]),
                    shadow=True,
                )

        if self.domain_randomize:
            # Visual property randomize
            if not self.canonical:
                # ground
                # print(len(self.room_ground.get_visual_bodies()[0].get_render_shapes()[0].material.base_color))
                for rs in self.room_ground.get_visual_bodies()[0].get_render_shapes():
                    apply_random_texture(rs.material, self.canonical_random)

                # walls
                for wall in self.walls:
                    for rs in wall.get_visual_bodies()[0].get_render_shapes():
                        apply_random_texture(rs.material, self.canonical_random)

                # tables
                for table in self.tables:
                    table_leg_materials = []

                    for body in table.get_visual_bodies():
                        for rs in body.get_render_shapes():
                            if body.name in ["upper_surface", "bottom_surface"]:
                                apply_random_texture(rs.material, self.canonical_random)
                            else:
                                table_leg_materials.append(rs.material)
                    apply_random_texture(table_leg_materials, self.canonical_random)

                # tray
                for body in self.tray.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        apply_random_texture(rs.material, self.canonical_random)

                # robot
                for link_idx, link in enumerate(self.robot.get_links()):
                    vb_list = link.get_visual_bodies()
                    for vb in vb_list:
                        for rs in vb.get_render_shapes():
                            apply_random_texture(rs.material, self.canonical_random)

            else:
                # ground
                self.room_ground.get_visual_bodies()[0].get_render_shapes()[0].material.set_base_color(
                    np.array(
                        [0.3, 0.3, 0.3, 1.0]
                    )
                )

                # walls
                for wall in self.walls:
                    for rs in wall.get_visual_bodies()[0].get_render_shapes():
                        rs.material.set_base_color(
                            np.array(
                                [0.5, 0.5, 0.5, 1.0]
                            )
                        )

                # tables
                for table in self.tables:
                    table_leg_materials = []
                    for body in table.get_visual_bodies():
                        for rs in body.get_render_shapes():
                            if body.name in ["upper_surface", "bottom_surface"]:
                                rs.material.set_base_color(
                                    np.array(
                                        # [0.6, 0.6, 0.6, 1.0]
                                        [0.1, 0.3, 0.1, 1.0]
                                    )
                                )
                            else:
                                rs.material.set_base_color(
                                    np.array(
                                        [0.6, 0.6, 0.6, 1.0]
                                    )
                                )


        if self.domain_randomize:
            pos_rand_range = (-0.05, 0.05)
            rot_rand_range = (-0.1, 0.1)
            fov_rand_range = (-0.05, 0.05)
            self.cameras["third"].set_pose(
                sapien.Pose(
                    self.standard_head_cam_pose.p
                    + self.np_random.uniform(*pos_rand_range, size=(3,)),
                    euler2quat(
                        quat2euler(self.standard_head_cam_pose.q)
                        + self.np_random.uniform(*rot_rand_range, size=(3,))
                    ),
                )
            )
            self.cameras["third"].set_fovx(
                self.standard_head_cam_fovx + self.np_random.uniform(*fov_rand_range),
                compute_y=True,
            )

        init_p = self.np_random.uniform(
            low=np.array([-0.01, -0.02, 0.78]), high=np.array([0.01, 0.02, 0.78])
        )

        init_angle = self.np_random.uniform(
            low=-0.01, high=0.01
        )

        self.robot.set_root_pose(
            sapien.Pose(
                init_p,
                np.array(
                    [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]
                ),
            )
        )
        new_dofs = self.robot.get_qpos().copy()

        new_dofs[self.arm_controller.arm_joint_indices] = (self.reset_joint_values
                                                           + self.np_random.uniform(-0.1, 0.1, size=(7,)))
        new_dofs[self.arm_controller.finger_joint_indices] = self.gripper_limit
        self.robot.set_qpos(new_dofs)
        self.robot.set_qvel(np.zeros_like(new_dofs))
        self.robot.set_qacc(np.zeros_like(new_dofs))

        self.init_base_pose = self._get_base_pose()
        # print("In reset, init_base_pose", self.init_base_pose)
        # reset stage for expert policy
        self.expert_phase = 0
        self.reset_tcp_pose = self._get_tcp_pose()

        # reload objects  TODO
        self.reload_objs(obj_list=obj_list)

        # if self.domain_randomize:
        #     for obj_id, obj_info in self.objs.items():
        #         obj = obj_info["actor"]
        #         for body in obj.get_visual_bodies():
        #             for rs in body.get_render_shapes():
        #                 # rs.material.set_base_color(np.concatenate([[0.7, 0.7, 0.7], [0.0]]))
        #                 if self.canonical:
        #                     rs.material.set_diffuse_texture_from_file("../../asset/pure_color_texture/proce_white.jpg")  #/white_07.jpg
        #                 else:
        #                     apply_random_texture(rs.material, self.canonical_random)

        self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]

        return obs, {}

    def step(self, action: np.ndarray):
        action = action.copy()
        info = {}

        if len(action) == 7:
            # action is desired base w, desired base v, delta eef pose, binary gripper action in the specified frame
            cur_ee_pose = self._get_tcp_pose()
            cur_relative_ee_pose = self.init_base_pose.inv().transform(cur_ee_pose)
            # action relative to tool frame
            if self.action_relative == "tool":
                desired_relative_ee_pose = cur_relative_ee_pose.transform(
                    sapien.Pose(
                        p=action[:3],
                        q=euler2quat(action[3:6]),
                    )
                )
            # action relative to fixed frame
            elif self.action_relative == "base":
                desired_relative_ee_pose = sapien.Pose(
                    cur_relative_ee_pose.p + self.p_scale * action[:3],
                    euler2quat(
                        quat2euler(cur_relative_ee_pose.q)
                        + self.rot_scale * action[3:6]
                    ),
                )
                # pose in initial agv frame
            elif self.action_relative == "none":
                desired_relative_ee_pose = sapien.Pose(
                    p=action[:3], q=euler2quat(action[3:6])
                )
                # dirty
                # action[8] = -action[8]
            desired_ee_pose = self.init_base_pose.transform(desired_relative_ee_pose)

            target_qpos = self.arm_controller.compute_q_target(
                desired_ee_pose, action[6]
            )
            info["desired_relative_pose"] = np.concatenate(
                [desired_relative_ee_pose.p, desired_relative_ee_pose.q]
            )
        else:
            assert len(action) == 8
            target_arm_q = action[:7]
            target_gripper_q = action[8] * np.ones(2)
            target_qpos = np.concatenate(
                [target_arm_q, target_gripper_q]
            )
        info["desired_joints"] = target_qpos[
            self.arm_controller.arm_joint_indices
        ]
        info["desired_gripper_width"] = (
            target_qpos[self.arm_controller.finger_joint_indices[0]]
        )

        self.robot.set_drive_target(target_qpos)

        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        for i in range(self.frame_skip):
            self.scene.step()

        # TODO: obs, reward, info
        self._update_observation()  # step may skip update obs for faster rollout  ???
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]
        is_success = None
        reward = None

        info.update(
            {
                "is_success": is_success,
                "init_base_pose": self.init_base_pose,
            }
        )
        return obs, reward, False, False, info

    def expert_action(self, obj_id, goal_obj_pose=None, noise_scale=0.0):
        # phases: before pregrasp, to grasp, close gripper, rotate, pull open
        actor = self.objs[obj_id]["actor"]
        init_p = self.objs[obj_id]["init_p"]
        init_q = self.objs[obj_id]["init_q"]
        init_scale = self.objs[obj_id]["init_scale"]
        obj_z_max = self.obj_infos[obj_id]["bbox"]["max"][-1] * init_scale[-1]
        done = False

        obj_pose = actor.get_pose()
        obj_init_pose = sapien.Pose(p=init_p, q=init_q)

        obj_grasp_pose = obj_init_pose.transform(sapien.Pose(
            p=np.array([self.obj_infos[obj_id]["bbox"]["max"][0] * init_scale[0], 0, 0])
        ))
        obj_pre_grasp_pose = obj_grasp_pose.transform(sapien.Pose(
            p=np.array([0, 0, 0.1])
        ))

        tray_pose = self.tray.get_pose()
        obj_place_pose = sapien.Pose(
            p=tray_pose.p + np.array([0, 0, obj_z_max + 0.02]) + (obj_grasp_pose.p - obj_init_pose.p), q=obj_init_pose.q
        )
        obj_pre_place_pose = obj_place_pose.transform(sapien.Pose(
            p=np.array([0, 0, 0.1])
        ))

        obj_T_grasp = sapien.Pose.from_transformation_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        desired_grasp_pose: sapien.Pose = None
        desired_gripper_width = None

        def apply_noise_to_pose(pose):
            pose.set_p(
                pose.p + self.np_random.uniform(-0.01, 0.01, size=(3,)) * noise_scale
            )
            pose.set_q(
                euler2quat(
                    quat2euler(pose.q)
                    + self.np_random.uniform(-0.1, 0.1, size=(3,)) * noise_scale
                )
            )

        if self.expert_phase == 0:
            desired_grasp_pose = obj_pre_grasp_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)

            # randomize gripper width in phase 0
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 1:
            desired_grasp_pose = obj_grasp_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    self.gripper_limit
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 2:
            gripper_width = self._get_gripper_width()
            desired_grasp_pose = obj_grasp_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 3:
            gripper_width = self._get_gripper_width()

            desired_grasp_pose = obj_pre_grasp_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 4:
            gripper_width = self._get_gripper_width()

            desired_grasp_pose = obj_pre_place_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 5:
            gripper_width = self._get_gripper_width()
            desired_grasp_pose = obj_place_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 6:
            desired_grasp_pose = obj_place_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    self.gripper_limit
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 7:
            desired_grasp_pose = self.reset_tcp_pose
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        else:
            raise NotImplementedError
        # TODO: error recovery
        tcp_pose = self._get_tcp_pose()

        if self.expert_phase == 0:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 1
        elif self.expert_phase == 1:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 2
        elif self.expert_phase == 2:
            if self._is_grasp(actor=actor, both_finger=True):
                self.expert_phase = 3

        elif self.expert_phase == 3:
            # print(tcp_pose.p, desired_grasp_pose.p)
            # print(tcp_pose.q, desired_grasp_pose.q)
            # print(np.linalg.norm(tcp_pose.p - desired_grasp_pose.p))
            # print(abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]))
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 4

        elif self.expert_phase == 4:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 5
        elif self.expert_phase == 5:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 6

        elif self.expert_phase == 6:
            # pass
            if not self._is_grasp(actor):  # lost grasp, need to regrasp
                self.expert_phase = 7
        elif self.expert_phase == 7:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 0
                done = True

        return action, done, {"desired_grasp_pose": desired_grasp_pose,
                              "desired_gripper_width": desired_gripper_width}

    # compute all the observations
    def _update_observation(self):
        self.observation_dict.clear()
        image_obs = self.capture_images_new()
        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_base_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        arm_joints = self.robot.get_qpos()[self.arm_controller.arm_joint_indices]
        self.observation_dict.update(image_obs)
        self.observation_dict["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        self.observation_dict["gripper_width"] = gripper_width
        self.observation_dict["robot_joints"] = arm_joints
        self.observation_dict["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width]
            ]
        )

    def get_observation(self, use_image=True):
        obs = dict()
        if use_image:
            image_obs = self.capture_images_new()
            obs.update(image_obs)

        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_base_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        arm_joints = self.robot.get_qpos()[self.arm_controller.arm_joint_indices]
        obs["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        obs["gripper_width"] = gripper_width
        obs["robot_joints"] = arm_joints
        obs["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width],
            ]
        )
        return obs

    def _get_tcp_pose(self) -> sapien.Pose:
        """
        return tcp pose in world frame
        """
        if self.tcp_link_idx is None:
            for link_idx, link in enumerate(self.robot.get_links()):
                if link.get_name() == "panda_grasptarget":
                    self.tcp_link_idx = link_idx
        link = self.robot.get_links()[self.tcp_link_idx]
        return link.get_pose()

    def _get_gripper_width(self) -> float:
        qpos = self.robot.get_qpos()
        return qpos[-2]

    def _get_base_pose(self) -> sapien.Pose:
        return self.robot.get_pose()

    def _desired_tcp_to_action(
            self,
            tcp_pose: sapien.Pose,
            gripper_width: float,
    ) -> np.ndarray:
        assert self.action_relative != "none"

        cur_tcp_pose = self._get_tcp_pose()
        cur_relative_tcp_pose = self.init_base_pose.inv().transform(cur_tcp_pose)
        desired_relative_tcp_pose = self.init_base_pose.inv().transform(tcp_pose)
        # print("get tcp pose", cur_tcp_pose, "desired tcp pose", tcp_pose)
        # relative to tool frame
        if self.action_relative == "tool":
            curtcp_T_desiredtcp = cur_relative_tcp_pose.inv().transform(
                desired_relative_tcp_pose
            )
            delta_pos = (
                    np.clip(curtcp_T_desiredtcp.p / self.p_scale, -1.0, 1.0)
                    * self.p_scale
            )
            delta_euler = (
                    np.clip(
                        wrap_to_pi(quat2euler(curtcp_T_desiredtcp.q)) / self.rot_scale,
                        -1.0,
                        1.0,
                    )
                    * self.rot_scale
            )
        # relative to fixed frame
        else:
            delta_pos = (
                    np.clip(
                        (desired_relative_tcp_pose.p - cur_relative_tcp_pose.p)
                        / self.p_scale,
                        -1.0,
                        1.0,
                    )
                    * self.p_scale
            )
            delta_euler = (
                    np.clip(
                        wrap_to_pi(
                            quat2euler(desired_relative_tcp_pose.q)
                            - quat2euler(cur_relative_tcp_pose.q)
                        )
                        / self.rot_scale,
                        -1.0,
                        1.0,
                    )
                    * self.rot_scale
            )
        return np.concatenate(
            [
                delta_pos,
                delta_euler,
                [gripper_width],
            ]
        )

    def _is_grasp(self, actor, threshold: float = 1e-4, both_finger=False):
        all_contact = self.scene.get_contacts()
        robot_finger_links: List[sapien.LinkBase] = [
            self.robot.get_links()[i] for i in self.finger_link_idxs
        ]
        finger_impulses = [
            get_pairwise_contact_impulse(
                all_contact, robot_finger, actor, None, None
            )
            for robot_finger in robot_finger_links
        ]
        finger_transforms = [
            robot_finger.get_pose().to_transformation_matrix()
            for robot_finger in robot_finger_links
        ]
        left_project_impulse = np.dot(finger_impulses[0], finger_transforms[0][:3, 1])
        right_project_impulse = np.dot(finger_impulses[1], -finger_transforms[1][:3, 1])
        # print(left_project_impulse, right_project_impulse)
        if both_finger:
            return (
                    left_project_impulse > threshold and right_project_impulse > threshold
            )
        else:
            return left_project_impulse > threshold or right_project_impulse > threshold

def test_expert_grasp():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        allow_dir=["side"] 
    )

    env = cano_pick_env
    cameras = ["third"]

    num_seeds = 1  # cano test
    num_vid = 10

    num_suc = 0

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(f"tmp/side", f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)
        env.reset(seed=seed)

        success_list = []

        video_writer = {cam: imageio.get_writer(
            f"tmp/seed_{seed}_{model_id[1]}.mp4",
            fps=20,
            format="FFMPEG",
            codec="h264",
        ) for cam in cameras}

        success = False
        frame_id = 0

        try:
            model_id = list(env.objs.keys())[0]
            prev_privileged_obs = None
            for step in range(500):
                action, done, desired_dict = env.expert_action(
                    noise_scale=0.5, obj_id=model_id,
                )
                _, _, _, _, info = env.step(action)

                if step < 500:
                    if done:
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

                    for cam in cameras:
                        image = obs.pop(f"{cam}-rgb")
                        video_writer[cam].append_data(image)
                    frame_id += 1

                else:
                    if done:
                        break
                    env.expert_phase = 6
        except Exception as e:
            print(seed, e)

        if success:
            success_list.append((seed, "s", frame_id))
            num_suc += 1
        else:
            success_list.append((seed, "f", frame_id))

        if seed < num_vid:
            for writer in video_writer.values():
                writer.close()

    print(num_suc)

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="none",
        allow_dir=["side"]
    )
    obs = env.get_observation()

    imageio.imwrite(os.path.join("tmp", f"test1.jpg"), obs[f"third-rgb"])
    # imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"forth-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    action = np.array([0.3, 0, 0.3, np.pi, 0, 0, 0.04])

    for _ in range(30):
        env.step(action)

    obs = env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"third-rgb"])

    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    # white = np.ones_like(obs[f"third-rgb"])
    #
    # for i in range(11):
    #     w = np.around(white * (i / 10) * 255).astype(np.uint8)
    #     imageio.imwrite("asset/pure_color_texture/white_{:02d}.jpg".format(i), w)


if __name__ == "__main__":
    test_expert_grasp()
    # test()
