import dataclasses
import numpy as np
import os
import sapien.core as sapien
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

# from pathlib import Path
ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../../asset")


@dataclasses.dataclass
class DoorConfig:
    board_x: float = 0.045
    board_y: float = 0.88
    board_z: float = 3.0
    frame_z: float = 3.0
    left_wall_y: float = 0.2
    right_wall_y: float = 1.0
    back_wall_x: float = 1.0
    left_frame_y: float = 0.05
    right_frame_y: float = 0.05
    connector_r: float = 0.01
    connector_l: float = 0.02
    mount_r: float = 0.03
    mount_l: float = 0.01
    handle_x: float = 0.02
    handle_y: float = 0.14
    handle_z: float = 0.02
    lock_x: float = 0.01
    lock_y: float = 0.02
    lock_z: float = 0.01

    # pos
    connector_pos_y = 0.35
    connector_pos_z = 1.2
    lock_pos_y = 0.35
    lock_pos_z = 1.1

    # angle
    rad_to_open = np.pi / 12


def generate_rand_door_config(np_random: np.random.RandomState, use_real: False):
    door_config = DoorConfig()
    door_config.board_x = np_random.uniform(0.04, 0.05)
    door_config.board_y = np_random.uniform(0.86, 0.9)

    door_config.connector_l = np_random.uniform(0.015, 0.025)
    door_config.connector_r = np_random.uniform(0.005, 0.015)
    door_config.mount_r = np_random.uniform(0.02, 0.04)
    door_config.mount_l = np_random.uniform(0.005, 0.015)
    door_config.handle_x = np_random.uniform(0.015, 0.025)
    door_config.handle_y = np_random.uniform(0.13, 0.15)
    door_config.handle_z = np_random.uniform(0.015, 0.025)
    door_config.lock_x = np_random.uniform(0.005, 0.015)
    door_config.lock_y = np_random.uniform(0.015, 0.025)
    door_config.lock_z = np_random.uniform(0.005, 0.015)
    # pos
    door_config.connector_pos_y = np_random.uniform(0.3, 0.4)
    if use_real:
        door_config.connector_pos_z = np_random.uniform(0.9, 1.1)
    else:
        door_config.connector_pos_z = np_random.uniform(1.1, 1.3)
    door_config.lock_pos_y = door_config.connector_pos_y
    door_config.lock_pos_z = door_config.connector_pos_z - np_random.uniform(0.05, 0.15)
    return door_config


def load_lab_door(
        scene: sapien.Scene,
        config: DoorConfig,
        np_random: np.random.RandomState,
        need_door_shut=True,
):
    if need_door_shut:
        mimic_lock_y = (config.board_y / 2 - config.connector_pos_y) / np.cos(
            config.rad_to_open
        )
    else:
        mimic_lock_y = 0.001

    builder = scene.create_articulation_builder()
    door_board_size = np.array([config.board_x, config.board_y, config.board_z])
    left_frame_size = np.array([config.board_x, config.left_frame_y, config.frame_z])
    right_frame_size = np.array([config.board_x, config.right_frame_y, config.frame_z])
    left_wall_size = np.array([config.board_x, config.left_wall_y, config.board_z])
    right_wall_size = np.array(
        [config.board_x + config.back_wall_x, config.right_wall_y, config.board_z]
    )
    door_mimic_lock_size = np.array([0.005, mimic_lock_y, 0.005])
    back_wall_size = np.array([config.back_wall_x, config.left_wall_y, config.board_z])
    # mimic_lock_handle_connector_size = (connector_r, connector_l) # cylinder, radius and length
    mimic_lock_handle_connector_size = np.array(
        [config.connector_l, 2 * config.connector_r, 2 * config.connector_r]
    )
    # connector_mount_size = (mount_r, mount_l)  # cylinder
    connector_mount_size = np.array(
        [config.mount_l, 2 * config.mount_r, 2 * config.mount_r]
    )
    handle_size = np.array([config.handle_x, config.handle_y, config.handle_z])
    # door_lock_mount_size = (mount_r, mount_l)
    door_lock_mount_size = np.array(
        [config.mount_l, 2 * config.mount_r, 2 * config.mount_r]
    )
    door_lock_size = np.array([config.lock_x, config.lock_y, config.lock_z])

    door_board_pos = np.array([0.0, 0.0, door_board_size[2] / 2])
    left_frame_pos = np.array(
        [0.0, door_board_size[1] / 2 + left_frame_size[1] / 2, left_frame_size[2] / 2]
    )
    right_frame_pos = np.array(
        [
            0.0,
            -door_board_size[1] / 2 - right_frame_size[1] / 2,
            right_frame_size[2] / 2,
        ]
    )
    left_wall_pos = np.array(
        [
            0.0,
            door_board_size[1] / 2 + left_frame_size[1] + left_wall_size[1] / 2,
            left_wall_size[2] / 2,
        ]
    )
    right_wall_pos = np.array(
        [
            right_wall_size[0] / 2 - door_board_size[0] / 2,
            -door_board_size[1] / 2 - right_frame_size[1] - right_wall_size[1] / 2,
            right_wall_size[2] / 2,
        ]
    )
    door_mimic_lock_pos = np.array(
        [
            door_board_size[0] / 2 + door_mimic_lock_size[0] / 2 + 0.001,
            config.connector_pos_y + door_mimic_lock_size[1] / 2,
            config.connector_pos_z,
        ]
    )
    back_wall_pos = np.array(
        [
            door_board_size[0] / 2
            + door_mimic_lock_size[0]
            + back_wall_size[0] / 2
            + 0.01,
            door_board_size[1] / 2 + left_wall_size[1] / 2,
            back_wall_size[2] / 2,
        ]
    )
    connector_pos = np.array(
        [
            -door_board_size[0] / 2
            - connector_mount_size[0]
            - mimic_lock_handle_connector_size[0] / 2,
            config.connector_pos_y,
            config.connector_pos_z,
        ]
    )
    connector_mount_pos = np.array(
        [
            -door_board_size[0] / 2 - connector_mount_size[0] / 2,
            config.connector_pos_y,
            config.connector_pos_z,
        ]
    )
    handle_pos = np.array(
        [
            -door_board_size[0] / 2
            - connector_mount_size[0]
            - mimic_lock_handle_connector_size[0]
            - handle_size[0] / 2,
            config.connector_pos_y
            + mimic_lock_handle_connector_size[1]
            - handle_size[1] / 2,
            config.connector_pos_z,
        ]
    )
    door_lock_mount_pos = np.array(
        [
            -door_board_size[0] / 2 - door_lock_mount_size[0] / 2,
            config.lock_pos_y,
            config.lock_pos_z,
        ]
    )
    door_lock_pos = np.array(
        [
            -door_board_size[0] / 2 - door_lock_mount_size[0] - door_lock_size[0] / 2,
            config.lock_pos_y,
            config.lock_pos_z,
        ]
    )

    wall_frame = builder.create_link_builder()
    wall_frame.set_name("wall")
    # left front wall
    wall_frame.add_box_collision(sapien.Pose(left_wall_pos), left_wall_size / 2)
    wall_frame.add_box_visual(sapien.Pose(left_wall_pos), left_wall_size / 2)
    # left back wall
    wall_frame.add_box_collision(sapien.Pose(back_wall_pos), back_wall_size / 2)
    wall_frame.add_box_visual(sapien.Pose(back_wall_pos), back_wall_size / 2)
    # right wall
    wall_frame.add_box_collision(sapien.Pose(right_wall_pos), right_wall_size / 2)
    wall_frame.add_box_visual(sapien.Pose(right_wall_pos), right_wall_size / 2)
    door_board_color_hsv = rgb_to_hsv(np.array([0.973, 0.757, 0.478]))
    door_board_color = hsv_to_rgb(
        np.array(
            [
                door_board_color_hsv[0] + np_random.uniform(-0.05, 0.05),
                door_board_color_hsv[1],
                door_board_color_hsv[2],
            ]
        )
    )
    # left frame
    wall_frame.add_box_collision(sapien.Pose(left_frame_pos), left_frame_size / 2)
    wall_frame.add_box_visual(
        sapien.Pose(left_frame_pos),
        left_frame_size / 2,
        color=door_board_color,
    )
    # right frame
    wall_frame.add_box_collision(sapien.Pose(right_frame_pos), right_frame_size / 2)
    wall_frame.add_box_visual(
        sapien.Pose(right_frame_pos),
        right_frame_size / 2,
        color=door_board_color,
    )

    board_frame = builder.create_link_builder(wall_frame)
    board_frame.set_name("door_board")
    hinge_joint_pose = sapien.Pose(
        np.array([-door_board_size[0] / 2, -door_board_size[1] / 2, 0.0]),
        np.array([np.cos(-np.pi / 4), 0.0, np.sin(-np.pi / 4), 0.0]),
    )
    board_frame.set_joint_properties(
        "revolute",
        limits=np.array([[0.0, np.pi / 2]]),
        pose_in_parent=hinge_joint_pose,
        pose_in_child=hinge_joint_pose,
        friction=0.001,
    )
    board_frame.add_box_collision(
        sapien.Pose(door_board_pos), door_board_size / 2, density=100
    )
    board_frame.add_box_visual(
        sapien.Pose(door_board_pos), door_board_size / 2, color=door_board_color
    )
    # connector mount
    # TODO: capsule is wierd
    # board_frame.add_capsule_collision(sapien.Pose(connector_mount_pos), connector_mount_size[0], connector_mount_size[1] / 2)
    # board_frame.add_capsule_visual(sapien.Pose(connector_mount_pos), connector_mount_size[0], connector_mount_size[1] / 2)
    board_frame.add_box_collision(
        sapien.Pose(connector_mount_pos), connector_mount_size / 2
    )
    board_frame.add_box_visual(
        sapien.Pose(connector_mount_pos), connector_mount_size / 2
    )

    # lock mount
    # board_frame.add_capsule_collision(sapien.Pose(door_lock_mount_pos), door_lock_mount_size[0], door_lock_mount_size[1] / 2)
    # board_frame.add_capsule_visual(sapien.Pose(door_lock_mount_pos), door_lock_mount_size[0], door_lock_mount_size[1] / 2)
    board_frame.add_box_collision(
        sapien.Pose(door_lock_mount_pos), door_lock_mount_size / 2
    )
    board_frame.add_box_visual(
        sapien.Pose(door_lock_mount_pos), door_lock_mount_size / 2
    )

    handle_frame = builder.create_link_builder(board_frame)
    handle_frame.set_name("door_handle")
    handle_joint_pose = sapien.Pose(connector_pos)
    handle_frame.set_joint_properties(
        "revolute",
        limits=np.array([[0.0, np.pi / 2]]),
        pose_in_parent=handle_joint_pose,
        pose_in_child=handle_joint_pose,
        friction=0.001,
    )
    # connector
    # handle_frame.add_capsule_collision(sapien.Pose(connector_pos), mimic_lock_handle_connector_size[0], mimic_lock_handle_connector_size[1] / 2)
    # handle_frame.add_capsule_visual(sapien.Pose(connector_pos), mimic_lock_handle_connector_size[0], mimic_lock_handle_connector_size[1] / 2)
    handle_frame.add_box_collision(
        sapien.Pose(connector_pos), mimic_lock_handle_connector_size / 2
    )
    handle_frame.add_box_visual(
        sapien.Pose(connector_pos), mimic_lock_handle_connector_size / 2
    )
    # handler
    rough_material = scene.create_physical_material(
        static_friction=2.0, dynamic_friction=2.0, restitution=0.0
    )
    handle_frame.add_box_collision(
        sapien.Pose(handle_pos),
        handle_size / 2,
        # material=rough_material
    )
    handle_frame.add_box_visual(sapien.Pose(handle_pos), handle_size / 2)
    # mimic lock (should be invisible)
    handle_frame.add_box_collision(
        sapien.Pose(door_mimic_lock_pos), door_mimic_lock_size / 2
    )
    handle_frame.add_box_visual(
        sapien.Pose(door_mimic_lock_pos),
        door_mimic_lock_size / 2,
        color=np.array([1.0, 0.0, 0.0]),
    )

    lock_frame = builder.create_link_builder(board_frame)
    lock_frame.set_name("door_lock")
    lock_joint_pose = sapien.Pose(door_lock_pos)
    lock_frame.set_joint_properties(
        "revolute",
        limits=np.array([[-np.pi / 2, np.pi / 2]]),
        pose_in_parent=lock_joint_pose,
        pose_in_child=lock_joint_pose,
    )
    # lock
    lock_frame.add_box_collision(sapien.Pose(door_lock_pos), door_lock_size / 2)
    lock_frame.add_box_visual(sapien.Pose(door_lock_pos), door_lock_size / 2)

    door_articulation = builder.build(fix_root_link=True)
    door_joint, handle_joint = door_articulation.get_active_joints()[:2]
    for link in door_articulation.get_links():
        if link.get_name() == "door_handle":
            rs_list = link.get_visual_bodies()[1].get_render_shapes()
            for rs in rs_list:
                rs.material.set_roughness(0.005)
                rs.material.set_metallic(0.5)
        elif link.get_name() == "door_lock":
            rs_list = link.get_visual_bodies()[0].get_render_shapes()
            for rs in rs_list:
                rs.material.set_roughness(0.005)
                rs.material.set_metallic(0.5)
        elif link.get_name() == "door_board":
            assert len(link.get_visual_bodies()) == 3
            for i in [1, 2]:
                rs_list = link.get_visual_bodies()[i].get_render_shapes()
                for rs in rs_list:
                    # rs.material.set_specular(0.5)
                    rs.material.set_roughness(0.005)
                    rs.material.set_metallic(0.5)
    door_joint.set_drive_property(0, 1, 5)
    handle_joint.set_drive_property(1, 0, 10)
    return door_articulation


def load_lab_wall(scene: sapien.Scene, position, length, angle=0., color=np.array([0.9, 0.9, 0.9])):
    wall_builder = scene.create_actor_builder()
    wall_builder.add_box_collision(
        sapien.Pose(p=np.array([0, 0, 3.0])),
        half_size=np.array([0.1, length / 2, 3.0]),
    )
    wall_builder.add_box_visual(
        sapien.Pose(p=np.array([0, 0, 3.0])),
        half_size=np.array([0.1, length / 2, 3.0]),
        color=color,
    )
    room_wall1 = wall_builder.build("room_wall_1")
    room_wall1.set_pose(
        sapien.Pose(
            np.array([position[0], position[1], 0.0]),
            np.array(
                [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]  # (w, x, y, z)
            ),
        )
    )
    room_wall1.lock_motion()

    return room_wall1


def load_table_4(
        scene: sapien.Scene,
        surface_size=np.array([2.0, 1.0, 0.03]),
        leg_size=np.array([0.04, 0.04, 0.73]),
        leg_pos_x=0.9,
        leg_pos_y=0.4,
        root_position=np.array([1.0, 0.0, 0.0]),
        root_angle=0
):
    """
    The origin is at the middle of the bottom of the four legs
    """
    table_builder = scene.create_actor_builder()
    # legs

    leg_pos_z = leg_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    # upper surface
    # surface_size = np.array([1.0, 1.6, 0.05])
    surface_pos_x = 0.0
    surface_pos_y = 0.0
    surface_pos_z = leg_size[2] + surface_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
        color=np.array([0., 0.4, 0.]),
        name="upper_surface"
    )
    table = table_builder.build(name="table")
    table.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    table.lock_motion()
    return table


def load_table_2(
        scene: sapien.Scene,
        surface_size=np.array([1.6, 0.8, 0.05]),
        leg_size=np.array([0.05, 0.1, 1.0]),
        leg_pos_x=0.7,
        foot_size=np.array([0.08, 0.6, 0.05]),
        root_position=np.array([1.5, 0.0, 0.0]),
        root_angle=0.
):
    """
    The origin is at the middle of the bottom of the four legs
    """
    table_builder = scene.create_actor_builder()
    # foots
    foot_pos_z = foot_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, 0, foot_pos_z])),
        half_size=foot_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, 0, foot_pos_z])),
        half_size=foot_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, 0, foot_pos_z])),
        half_size=foot_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, 0, foot_pos_z])),
        half_size=foot_size / 2,
    )
    # legs
    leg_pos_z = leg_size[2] / 2 + foot_size[2]
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, 0, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, 0, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, 0, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, 0, leg_pos_z])),
        half_size=leg_size / 2,
    )
    # upper surface
    # surface_size = np.array([1.0, 1.6, 0.05])
    surface_pos_x = 0.0
    surface_pos_y = 0.0
    surface_pos_z = foot_size[2] + leg_size[2] + surface_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
        color=np.array([1, 0.5, 0.]),
        name="upper_surface"
    )
    table = table_builder.build(name="table_2")
    table.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    table.lock_motion()
    return table


def load_storage_box(
        scene: sapien.Scene,
        box_size=np.array([0.3, 0.3, 0.05]),
        thickness=0.005,
        # color=np.array([0.6, 0.3, 0]),
        color=np.array([1.0, 1.0, 1.0]),
        root_position=np.array([1., 0.0, 0.0]),
        root_angle=0.
):
    box_builder = scene.create_actor_builder()

    # bottom
    box_builder.add_box_collision(
        sapien.Pose(p=np.array([0, 0, 0])),
        half_size=np.array([box_size[0], box_size[1], thickness]) / 2,
    )
    box_builder.add_box_visual(
        sapien.Pose(p=np.array([0, 0, 0])),
        half_size=np.array([box_size[0], box_size[1], thickness]) / 2,
        color=color/ 2,
        name="bottom_surface"
    )
    # sides
    side_x = box_size[0] / 2 - thickness / 2
    side_y = box_size[1] / 2 - thickness / 2
    side_z = thickness + box_size[2] / 2
    box_builder.add_box_collision(
        sapien.Pose(p=np.array([side_x, 0, side_z])),
        half_size=np.array([thickness, box_size[1], box_size[2]]) / 2,
    )
    box_builder.add_box_visual(
        sapien.Pose(p=np.array([side_x, 0, side_z])),
        half_size=np.array([thickness, box_size[1], box_size[2]]) / 2,
        color=color
    )
    box_builder.add_box_collision(
        sapien.Pose(p=np.array([-side_x, 0, side_z])),
        half_size=np.array([thickness, box_size[1], box_size[2]]) / 2,
    )
    box_builder.add_box_visual(
        sapien.Pose(p=np.array([-side_x, 0, side_z])),
        half_size=np.array([thickness, box_size[1], box_size[2]]) / 2,
        color=color
    )
    box_builder.add_box_collision(
        sapien.Pose(p=np.array([0, side_y, side_z])),
        half_size=np.array([box_size[0], thickness, box_size[2]]) / 2,
    )
    box_builder.add_box_visual(
        sapien.Pose(p=np.array([0, side_y, side_z])),
        half_size=np.array([box_size[0], thickness, box_size[2]]) / 2,
        color=color
    )
    box_builder.add_box_collision(
        sapien.Pose(p=np.array([0, -side_y, side_z])),
        half_size=np.array([box_size[0], thickness, box_size[2]]) / 2,
    )
    box_builder.add_box_visual(
        sapien.Pose(p=np.array([0, -side_y, side_z])),
        half_size=np.array([box_size[0], thickness, box_size[2]]) / 2,
        color=color
    )
    storage_box = box_builder.build(name="storage_box")
    storage_box.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    storage_box.lock_motion()
    return storage_box


def load_blocks_on_table(scene: sapien.Scene):
    blocks_builder = scene.create_actor_builder()

    block_size = np.array([0.15, 0.05, 0.05])

    blocks_builder.add_box_collision(
        sapien.Pose(p=np.array([0.5, 0, 1.0])),
        half_size=block_size / 2,
    )
    blocks_builder.add_box_visual(
        sapien.Pose(p=np.array([0.5, 0, 1.0])),
        half_size=block_size / 2,
        color=np.array([1, 0, 0])
    )
    block = blocks_builder.build(name="block")
    return block


def load_lab_scene_urdf(scene: sapien.Scene):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    urdf_path = os.path.join(
        os.path.dirname(__file__), "../../../asset/2004/mobility_cvx.urdf"
    )
    door_articulation = loader.load(
        urdf_path,
    )
    for link in door_articulation.get_links():
        if link.get_name() == "link_switch_and_lock":
            for visualbodies in link.get_visual_bodies():
                if visualbodies.get_name() == "lock_bar":
                    lock_bar_pose_p = visualbodies.local_pose.p
                    break
            for s in link.get_collision_shapes():
                if (lock_bar_pose_p == s.get_local_pose().p).all():
                    s.set_collision_groups(2, 2, 0, 0)
                    lock_bar_pose_p = 0
                else:
                    s.set_collision_groups(3, 3, 1 << 31, 0)
        else:
            for s in link.get_collision_shapes():
                g0, g1, g2, g3 = s.get_collision_groups()
                s.set_collision_groups(3, 3, 1 << 31, 0)
    # for test
    # for link in door_articulation.get_links():
    #     print("linkname:",link.get_name())
    #     for s in link.get_collision_shapes():
    #         g0, g1, g2, g3 = s.get_collision_groups()
    #         print("door after:g0,g1,g2,g3",hex(g0),hex(g1), hex(g2), hex(g3))
    door_joint, handle_joint = door_articulation.get_active_joints()[:2]
    door_joint.set_drive_property(0, 1, 5)
    handle_joint.set_drive_property(1, 0, 10)
    return door_articulation


def build_actor_ycb(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        root_dir=os.path.join(ASSET_DIR, "mani_skill2_ycb"),
        root_position=np.array([1.0, 1.0, 1.5]),
        root_angle=0.,
        obj_allow_dir = "along"
):
    builder = scene.create_actor_builder()
    model_dir = os.path.join(root_dir, "models", model_id)

    collision_file = os.path.join(model_dir, "collision.obj")
    if not os.path.exists(collision_file):
        collision_file = os.path.join(root_dir,"models", obj_allow_dir, model_id, "collision", "collision.obj")

    if isinstance(scale, float):
        scale = [scale] * 3

    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=scale,
        material=physical_material,
        density=density,
    )

    visual_file = os.path.join(model_dir, "textured.obj")
    if not os.path.exists(visual_file):
        visual_file = os.path.join(root_dir,"models", obj_allow_dir, model_id, "poisson", "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=scale)

    actor = builder.build(name=model_id)
    actor.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    return actor


def build_actor_egad(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        render_material: sapien.RenderMaterial = None,
        density=100,
        root_dir=os.path.join(ASSET_DIR, "mani_skill2_egad"),
        root_position=np.array([1.0, 1.0, 1.5]),
        root_angle=0.
):
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    # split = "train" if "_" in model_id else "eval"

    collision_file = os.path.join(root_dir, f"egad_train_set_coacd", f"{model_id}.obj")
    assert os.path.exists(collision_file)
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = os.path.join(root_dir, f"egad_train_set", f"{model_id}.obj")
    builder.add_visual_from_file(
        filename=visual_file, scale=[scale] * 3, material=render_material
    )

    actor = builder.build()
    actor.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    return actor


def build_actor_real(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        render_material: sapien.RenderMaterial = None,
        density=1000,
        root_dir=os.path.join(ASSET_DIR, "real_assets"),
        root_position=np.array([1.0, 1.0, 1.5]),
        root_rot = np.array([0., 0., 0., 1.]),
):

    builder = scene.create_actor_builder()
    collision_file = os.path.join(root_dir, "available", "collision", f"{model_id}_collision.obj")

    if isinstance(scale, (int, float)):
        scale = [scale] * 3
    print(scale)

    builder.add_multiple_collisions_from_file(   # without convex
        filename=collision_file,
        scale=scale,
        material=physical_material,
        density=density,
    )

    visual_file = os.path.join(root_dir,"available",  f"{model_id}.glb")  #

    builder.add_visual_from_file(
        filename=visual_file, scale=scale, material=render_material
    )

    actor = builder.build()
    actor.set_pose( sapien.Pose(root_position, root_rot) )

    return actor