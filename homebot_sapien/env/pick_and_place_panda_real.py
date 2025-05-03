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
from homebot_sapien.env.base import BaseEnv, recover_action, get_pairwise_contact_impulse #, get_pairwise_contacts

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from homebot_sapien.utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from homebot_sapien.env.utils import apply_random_texture,apply_cano_texture, check_intersect_2d, grasp_pose_process, check_intersect_2d_
from homebot_sapien.env.articulation.pick_and_place_articulation import (
    # generate_rand_door_config,
    load_lab_wall,
    load_table_4,
    load_storage_box,
    build_actor_ycb,
    build_actor_egad,
    build_actor_real,
    ASSET_DIR
)
from homebot_sapien.env.robot import load_robot_panda
from homebot_sapien.env.controller.whole_body_controller import ArmSimpleController
import json
import pickle
import requests
from datetime import datetime

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
        super().__init__(use_gui, device, mipmap_levels)

        ### new
        # cam_p = np.array([0.763579180, -0.03395012, 1.44071344])   
        # look_at_dir = np.array([-0.53301526,  0.01688062,  -0.84593722])  
        # right_dir = np.array([0.05021884, 0.99866954, -0.01171393])  
        ### origin 
        cam_p = np.array([0.793, -0.056, 1.505])   
        look_at_dir = np.array([-0.616, 0.044, -0.787])  
        right_dir = np.array([0.036, 0.999, 0.027])
        self.create_camera(
            position=cam_p,
            look_at_dir=look_at_dir,
            right_dir=right_dir,
            name="third",
            resolution=(320,240),
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
            [0., -0.85, 0., -2.8, 0., 2.1, 0.785]
        )
        self.joint_scale = (joint_high - joint_low) / 2
        self.joint_mean = (joint_high + joint_low) / 2
        # Set spaces
        ycb_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_ycb", "info_pick.json"), "r"))  # v0+v1
        egad_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_egad", "info_pick_train_v1.json"), "r"))
        real_models = json.load(open(os.path.join(ASSET_DIR, "real_assets", "info_pick_avail.json"), "r"))  # v0+v1+v2  # for debug

        self.model_db = dict(
            ycb=ycb_models,
            egad=egad_models,
            real=real_models
        )

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
        self.storage_box = load_storage_box(self.scene, root_position=np.array([0.4, -0.2, self.table_top_z]))
        # storage_box_materials = []
        # for body in self.storage_box.get_visual_bodies():
        #     for rs in body.get_render_shapes():
        #         if body.name in ["upper_surface", "bottom_surface"]:
        #             apply_cano_texture(rs.material)

        #         else:
        #             storage_box_materials.append(rs.material)
        # apply_cano_texture(storage_box_materials)


    def compute_init_pose(self, model_type, model_id, rand_p):
        bbox_min_z = self.model_db[model_type][model_id]["bbox"]["min"][-1] * \
                        self.model_db[model_type][model_id]["scales"][0]
        bbox_min_y = self.model_db[model_type][model_id]["bbox"]["min"][1] * \
                        self.model_db[model_type][model_id]["scales"][0]
        bbox_min_x = self.model_db[model_type][model_id]["bbox"]["min"][0] * \
                        self.model_db[model_type][model_id]["scales"][0]

        bbox_max_z = self.model_db[model_type][model_id]["bbox"]["max"][-1] * \
                        self.model_db[model_type][model_id]["scales"][0]
        bbox_max_y = self.model_db[model_type][model_id]["bbox"]["max"][1] * \
                        self.model_db[model_type][model_id]["scales"][0]
        bbox_max_x = self.model_db[model_type][model_id]["bbox"]["max"][0] * \
                        self.model_db[model_type][model_id]["scales"][0]
        
        offset_x = (bbox_max_x - bbox_min_x) / 2
        offset_y = (bbox_max_y - bbox_min_y) / 2
        offset_z = (bbox_max_z - bbox_min_z) / 2

        along = self.model_db[model_type][model_id]["along"]
        if model_type == "egad":
            allow_dir = []
        else:
            allow_dir = self.model_db[model_type][model_id]["allow_dir"]

        #  only trans for real column and side, without init_angle randomization
        allow_list = ["side","column"]
        if model_type == "real" and allow_dir in allow_list:
            init_angle = 0
            rand_q = np.array([np.cos(init_angle / 2), 0, 0, np.sin(init_angle / 2)])    
            # side, column must trans (side cannot support z trans yet)
            if allow_dir == "column":
                if  along == "z":
                    init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                        [0.4, 0.2, self.table_top_z -bbox_min_x + 5e-3])  
                    ##### change z (along z-axis) to z (along y-axis)
                    init_trans = "z_to_y"
                    # tran_q = np.array([0.5, 0.5, 0.5, 0.5])   # x90, y90
                    tran_q = np.array([0.5, 0.5, -0.5, 0.5])   # x90, z90                
                    init_q = qmult(rand_q, tran_q)
                elif along == "y":
                    init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                        [0.4, 0.2, self.table_top_z- bbox_min_y+ 5e-3])   
                    init_trans = "y_to_z"
                    tran_q = np.array([np.cos(np.pi/4), np.sin(np.pi/4),0, 0])   # x90  
                    init_q = qmult(rand_q, tran_q) 
                else:
                    init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                        [0.4, 0.2, self.table_top_z-bbox_min_z + 5e-3])
                    init_trans = None
                    init_q = rand_q
            elif allow_dir == "side":
                if along == "x" or along == "y":
                    init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                        [0.4, 0.2, self.table_top_z- bbox_min_y+ 5e-3])
                    init_trans = "y_to_z"
                    tran_q = np.array([np.cos(np.pi/4), np.sin(np.pi/4),0, 0])   # x90  
                    init_q = qmult(rand_q, tran_q) 
        else:
            init_angle = self.np_random.uniform(-np.pi, np.pi)  # for debug
            rand_q = np.array([np.cos(init_angle / 2), 0, 0, np.sin(init_angle / 2)])    
            init_trans = None
            init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                [0.4, 0.2, self.table_top_z-bbox_min_z + 5e-3])
            init_q = np.array( [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)] )

        return init_p, init_angle, init_q, init_trans, (offset_x, offset_y, offset_z)

    def reload_objs(self, obj_list=None, egad_ratio=0.5, num_obj=1):  
        if hasattr(self, "objs"):
            for obj_id, obj_dict in self.objs.items():
                self.scene.remove_actor(obj_dict["actor"])
            self.objs.clear()
        else:
            self.objs = dict()

        if obj_list is None:
            num_egad = int( (len(self.model_db["ycb"].keys())+len(self.model_db["real"].keys())) / (1 - egad_ratio) * egad_ratio)
            num_egad = min(num_egad, len(self.model_db["egad"].keys()))
            egad_list = self.np_random.choice(list(self.model_db["egad"].keys()), num_egad, replace=False)
            egad_list = [("egad", model_id) for model_id in egad_list]
            ycb_list = [("ycb", model_id) for model_id in self.model_db["ycb"].keys()]
            real_list = [("real", model_id) for model_id in self.model_db["real"].keys()]
            # egad_list+  # for debug [need] +   ycb_list
            obj_list = self.np_random.choice( ycb_list+egad_list+real_list , num_obj, replace=False)

        for model_type, model_id in obj_list:
            # model_id = "pepper_v1"  #for debug
            num_try = 0
            obj_invalid, init_p, init_angle, init_q, init_trans, obj = True, None, None, None, None, None
            if model_type == "egad":
                obj_allow_dir = None
            else:
                obj_allow_dir = self.model_db[model_type][model_id]["allow_dir"]
            # print("obj_allow_dir", obj_allow_dir)
            while num_try < 10 and obj_invalid:
                rand_p = self.np_random.uniform(-0.13, 0.13, size=(2,))    # 0.15
                init_p, init_angle, init_q, init_trans, offset = self.compute_init_pose(model_type, model_id, rand_p)

                obj_invalid = False
                if rand_p[0] < -0.05 and rand_p[1] < -0.05:
                    obj_invalid = True
                else:
                    for (prev_model_type, prev_model_id), prev_model in self.objs.items():
                        if check_intersect_2d_(init_p, self.model_db[model_type][model_id]["bbox"],
                                               self.model_db[model_type][model_id]["scales"][0],
                                               self.objs[(prev_model_type, prev_model_id)]["init_p"],
                                               self.model_db[prev_model_type][prev_model_id]["bbox"],
                                               self.model_db[prev_model_type][prev_model_id]["scales"][0]):
                            obj_invalid = True
                            break
                num_try += 1

            if not obj_invalid:
                if model_type == "egad":
                    mat = self.renderer.create_material()
                    color = self.np_random.uniform(0.2, 0.8, 3)
                    color = np.hstack([color, 1.0])
                    mat.set_base_color(color)
                    mat.metallic = 0.0
                    mat.roughness = 0.1
                    obj = build_actor_egad(
                        model_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=self.model_db[model_type][model_id]["density"] if "density" in
                                                                                  self.model_db[model_type][
                                                                                      model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0],
                        render_material=mat,
                    )
                elif model_type == "ycb":
                    obj = build_actor_ycb(
                        model_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=self.model_db[model_type][model_id]["density"] if "density" in
                                                                                  self.model_db[model_type][
                                                                                      model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0],
                        obj_allow_dir=obj_allow_dir
                    )
                    obj.set_damping(0.1, 0.1)
                elif model_type == "real":
                    obj = build_actor_real(
                        model_id, self.scene, root_position=init_p, root_rot=init_q,
                        density=self.model_db[model_type][model_id]["density"] if "density" in self.model_db[model_type][model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0]
                    )
                    obj.set_damping(0.1, 0.1)
                else:
                    raise Exception("unknown data type!")

                self.objs.update({(model_type, model_id): dict(actor=obj, init_p=init_p, init_q = init_q, init_trans=init_trans, offset=offset)})

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

        self.walls = [
            self.room_wall1,  # self.room_wall2,  # self.room_wall3,
        ]
        self.tables = [
            self.table4,  # self.table2,
        ]

    def reset(self, seed: int = None, options: dict = None, obj_list: List = None):
        super().reset(seed, options)
        self.canonical_random, _ = seeding.np_random(seed)

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            table_rand = self.np_random.uniform(-1, 1, (2,))
            table_rand_size = np.array([0.1 * table_rand[0], 0.3 * table_rand[1], 0.]) + np.array([1.0, 1.8, 0.03])
            leg_pos_x = table_rand_size[0] / 2 - 0.1
            leg_pos_y = table_rand_size[1] / 2 - 0.1
            table_position = np.array([0.35, 0, 0])
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

                # storage box
                storage_box_materials = []
                for body in self.storage_box.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        if body.name in ["upper_surface", "bottom_surface"]:
                            apply_random_texture(rs.material, self.canonical_random)
                        else:
                            storage_box_materials.append(rs.material)
                apply_random_texture(storage_box_materials, self.canonical_random)

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

                # storage box
                for body in self.storage_box.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        if body.name in ["upper_surface", "bottom_surface"]:
                            rs.material.set_base_color(
                                np.array(
                                    [0.4, 0.2, 0.0, 1.0]
                                )
                            )
                        else:
                            rs.material.set_base_color(
                                np.array(
                                    [0.6, 0.3, 0.0, 1.0]
                                )
                            )

                # for link_idx, link in enumerate(self.robot.get_links()):
                #     vb_list = link.get_visual_bodies()
                #     for vb in vb_list:
                #         for rs in vb.get_render_shapes():
                #             material = rs.material
                #             print(link, material.base_color)

        if self.domain_randomize:
            # Camera pose
            # print("Camera pose", self.cameras["third"].get_pose())
            # print("fov", self.cameras["third"].fovx, self.cameras["third"].fovy)
            # Randomize camera pose

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
            # 取消相机随机变换，直接设置为标准相机位姿与视角 for debug
            self.cameras["third"].set_pose(self.standard_head_cam_pose)
            self.cameras["third"].set_fovx(self.standard_head_cam_fovx, compute_y=True)
            # self.cameras["wrist"].set_fovx(
            #     self.standard_wrist_cam_fovx + self.np_random.uniform(-0.1, 0.1),
            #     compute_y=True,
            # )

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
        self.reload_objs(obj_list=obj_list)

        # TODO: get obs
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
        self._update_observation()  # zzt step may skip update obs for faster rollout
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

    def expert_pose_define(self, obj_id):
        init_p = self.objs[obj_id]["init_p"]
        init_q =  self.objs[obj_id]["init_q"]
        init_trans = self.objs[obj_id]["init_trans"]
        along = self.model_db[obj_id[0]][obj_id[1]]["along"]
        obj_z_max = self.model_db[obj_id[0]][obj_id[1]]["bbox"]["max"][-1] * \
                    self.model_db[obj_id[0]][obj_id[1]]["scales"][0]
        obj_z_min = self.model_db[obj_id[0]][obj_id[1]]["bbox"]["min"][-1] * \
                    self.model_db[obj_id[0]][obj_id[1]]["scales"][0]
        obj_y_max = self.model_db[obj_id[0]][obj_id[1]]["bbox"]["max"][1] * \
                    self.model_db[obj_id[0]][obj_id[1]]["scales"][0]
        obj_x_max = self.model_db[obj_id[0]][obj_id[1]]["bbox"]["max"][0] * \
                    self.model_db[obj_id[0]][obj_id[1]]["scales"][0]
        if obj_id[0] == 'egad':
            allow_dir = []
        else:
            allow_dir = self.model_db[obj_id[0]][obj_id[1]]["allow_dir"]
        
        # deal with real assets: y_min = 0
        offset_y = obj_y_max / 2 if obj_id[0] == 'real' else 0
        ### trans pose
        if init_trans == None:        
            # grasp along==x,y, cannot exceed gripper height
            offset_z = obj_z_max-0.035 if (obj_z_max > 0.035) else 0  
            if along == "x" or along == "z":
                obj_T_grasp = sapien.Pose.from_transformation_matrix(
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, -1, 0, offset_y],
                            [0, 0, -1, offset_z],
                            [0, 0, 0, 1],
                        ]
                    )
                )
            elif along == "y":
                obj_T_grasp = sapien.Pose.from_transformation_matrix(
                    np.array(
                        [
                            [0, 1, 0, 0],
                            [1, 0, 0, offset_y],  
                            [0, 0, -1,offset_z],
                            [0, 0, 0, 1],  
                        ]
                    )
                )
            # elif along == "z":   #(not that work)
            #     offset_z = max(0.04- np.abs(obj_z_min), (obj_z_min+obj_z_min) /2) if np.abs(obj_z_min) < 0.01 else 0
            #     rand = self.np_random.uniform(0,1)     
            #     if rand < 0.5:
            #         mat =  np.array([
            #                 # ori z1
            #                 [0, 0, 1, offset_x],
            #                 [0, 1, 0, offset_y],
            #                 [-1, 0, 0, offset_z], 
            #                 [0, 0, 0, 1] 
            #             ]
            #         )
            #     else:
            #         mat = np.array( [
            #                 # ori z2
            #                 [0, 0, 1, offset_x],
            #                 [0, 1, 0, offset_y],
            #                 [-1, 0, 0, offset_z], 
            #                 [0, 0, 0, 1]
            #             ]
            #         )  
            #     obj_T_grasp = sapien.Pose.from_transformation_matrix(mat) 

            obj_T_pregrasp = sapien.Pose(
                p=np.array([0.0, 0.0, 0.06 + obj_z_max]), q=obj_T_grasp.q  
            )
            obj_T_postgrasp = sapien.Pose(
                p=np.array([0.0, 0.0, 0.1 + obj_z_max]), q=obj_T_grasp.q
            )
            obj_T_pregoal = sapien.Pose(
                    p=np.array([0.0, 0.0, 0.1 + obj_z_max * 2]), q=obj_T_grasp.q
            )
        elif init_trans == "z_to_y":
            if allow_dir == "column":   ### only for umbrella
                # grasp along==x,y, cannot exceed gripper height
                offset_x = obj_x_max-0.035 if (obj_x_max > 0.035) else 0  
                ori_obj_T_grasp = sapien.Pose.from_transformation_matrix(
                    np.array(
                        [
                            # # ori y
                            [0, 1, 0, offset_x],
                            [1, 0, 0, offset_y],  
                            [0, 0, -1, 0],
                            [0, 0, 0, 1],  
                        ]
                    )
                )       
            else:
                raise Exception("unsupported allow_dir for z_to_y!")
            from transforms3d.quaternions import qconjugate 
            combined_q = qmult( qconjugate(init_q),ori_obj_T_grasp.q)
            obj_T_grasp = sapien.Pose(p=ori_obj_T_grasp.p, q=combined_q)
            
            # transform form object frame
            obj_T_pregrasp = sapien.Pose(
                p=np.array([0.06+ obj_x_max, 0.0, 0.0]), q=obj_T_grasp.q   
            )
            obj_T_postgrasp = sapien.Pose(
                p=np.array([0.1 + obj_x_max, 0.0, 0.0]), q=obj_T_grasp.q
            )
            # transform from world frame  
            obj_T_pregoal = sapien.Pose(
                    p=np.array([0.0, 0.0 , 0.1 + obj_x_max * 2]), q=ori_obj_T_grasp.q
            )   
        elif init_trans == "y_to_z":
            ###   grasp along==x,y, cannot exceed gripper height
            if obj_id[0] == 'real':    
                offset_y = obj_y_max - 0.035 if obj_y_max > 0.07 else obj_y_max / 2
            else:        
                offset_y = obj_y_max - 0.035 if obj_y_max > 0.035 else 0

            if allow_dir == "column":
                # column use z to grasp (not that work)
                # offset_z = max(0.04- np.abs(obj_z_min), (obj_z_min+obj_z_min) /2) if np.abs(obj_z_min) < 0.01 else 0
                # rand = self.np_random.uniform(0,1)     
                # if rand < 0.5:
                #     mat =  np.array([
                #             # ori z1
                #             [0, 0, 1, offset_x],
                #             [0, 1, 0, offset_y],
                #             [-1, 0, 0, offset_z], 
                #             [0, 0, 0, 1] 
                #         ]
                #     )
                # else:
                #     mat = np.array( [
                #             # ori z2
                #             [0, 0, 1, offset_x],
                #             [0, 1, 0, offset_y],
                #             [-1, 0, 0, offset_z], 
                #             [0, 0, 0, 1]
                #         ]
                #     )

                mat = np.array(
                    [
                        # ori x
                        [1, 0, 0, 0],
                        [0, -1, 0, offset_y],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1],
                    ]
                )
            elif allow_dir == "side":
                # side use x to grasp (already tested), and must grasp the 'wall' (offset x)
                mat = np.array(
                    [
                        # ori x 
                        [1, 0, 0, 0],  
                        [0, -1, 0, offset_y],
                        [0, 0, -1, self.gripper_limit ],
                        [0, 0, 0, 1],
                    ]
                )
            else:
                raise Exception("unsupported allow_dir for y_to_z!")   
            ori_obj_T_grasp = sapien.Pose.from_transformation_matrix(mat) 
            from transforms3d.quaternions import qconjugate 
            combined_q = qmult( qconjugate(init_q),ori_obj_T_grasp.q)
            obj_T_grasp = sapien.Pose(p=ori_obj_T_grasp.p, q=combined_q)

            # transform form object frame
            obj_T_pregrasp = sapien.Pose(
                p=np.array([0, 0.06+ obj_y_max, 0.0]), q=obj_T_grasp.q   
            )
            obj_T_postgrasp = sapien.Pose(
                p=np.array([0.0, 0.1 + obj_y_max, 0.0]), q=obj_T_grasp.q
            )
            # transform from world frame
            obj_T_pregoal = sapien.Pose(
                    p=np.array([0.0, 0.0 , 0.1 + obj_y_max]), q=ori_obj_T_grasp.q
            )
        else:
            raise Exception("unknown transform type!") 

        init_obj_pose = sapien.Pose( p=init_p, q=init_q  )

        return obj_T_grasp, obj_T_pregrasp, obj_T_postgrasp, obj_T_pregoal, init_obj_pose

    def expert_action(self, obj_id, goal_obj_pose, noise_scale=0.0):
        # phases: before pregrasp, to grasp, close gripper, rotate, pull open
        actor = self.objs[obj_id]["actor"]
        obj_pose = actor.get_pose()
        obj_T_grasp, obj_T_pregrasp, obj_T_postgrasp, obj_T_pregoal, init_obj_pose = self.expert_pose_define(obj_id)

        desired_grasp_pose: sapien.Pose = None
        desired_gripper_width = None
        done = False

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

        # print("expert_phase", self.expert_phase) # for debug
        if self.expert_phase == 0:
            desired_grasp_pose = init_obj_pose.transform(obj_T_pregrasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)

            # randomize gripper width in phase 0
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit) #  self.gripper_limit

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 1:
            desired_grasp_pose = init_obj_pose.transform(obj_T_grasp)
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
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
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

            desired_grasp_pose = init_obj_pose.transform(obj_T_postgrasp)
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
            #### transform from goal(world)
            desired_grasp_pose = goal_obj_pose.transform(obj_T_pregoal)
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
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
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
        elif self.expert_phase == 6:
            desired_grasp_pose = self.reset_tcp_pose
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
        else:
            raise NotImplementedError
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
            # pass
            if not self._is_grasp(actor):  # lost grasp, need to regrasp
                self.expert_phase = 6
        elif self.expert_phase == 6:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.05  # 0.01
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
        # self.observation_dict["door_states"] = door_states
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
        # if self.agv_link_idx is None:
        #     for link_idx, link in enumerate(self.robot.get_links()):
        #         if link.get_name() == "mount_link_fixed":
        #             self.agv_link_idx = link_idx
        # link = self.robot.get_links()[self.agv_link_idx]
        # return link.get_pose()

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

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="none"
    )
    env.reset(seed=200)
    obs = env.get_observation()

    imageio.imwrite(os.path.join("tmp", f"test1.jpg"), obs[f"third-rgb"])
    # imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"forth-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    action = np.array([0.3, 0, 0.3, np.pi, 0, 0, 0.04])

    for i in range(30):
        env.step(action)

    obs = env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"third-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

def test_expert_grasp():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # from homebot_sapien.env.pick_and_place_panda import PickAndPlaceEnv
    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True
    )

    env = cano_pick_env
    cameras = ["third"]

    num_seeds = 1  # cano test
    num_vid = 10

    num_suc = 0

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):

        seed = 6 #200
        env.reset(seed=seed)
        random_state = np.random.RandomState(seed=seed)
        model_id_list = list(env.objs.keys())
        random_state.shuffle(model_id_list)

        success_list = []

        for ep_id, model_id in enumerate(model_id_list):

            # if seed < num_vid:
            video_writer = {cam: imageio.get_writer(
                # f"tmp/seed_{seed}_ep_{ep_id}_cam_{cam}.mp4",
                f"tmp/seed_{seed}_{model_id[1]}.mp4",
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            success = False
            frame_id = 0

            try:
                goal_p_rand = 0 # random_state.uniform(-0.1, 0.1, size=(2,))
                goal_q_rand = 0 # random_state.uniform(-0.5, 0.5)

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

                    if step < 500:
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

                        for cam in cameras:
                            image = obs.pop(f"{cam}-rgb")
                            # if seed < num_vid:
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
                num_suc += 1
            else:
                success_list.append((ep_id, "f", frame_id))

            if seed < num_vid:
                for writer in video_writer.values():
                    writer.close()

    print(num_suc)

if __name__ == "__main__":
    # test()
    test_expert_grasp()
