
# homebot_sapien code

## /env

- scene objects and task environment definition

### Articulation Define

- door_articulation
- drawer_articulation
- pick_and_place_articulation

### Env Define 

- open_door: 
    - create own env OpenDoor (only registered)
    - from .door_articulation import (
        load_lab_door,
        generate_rand_door_config,
        load_lab_wall,
        load_lab_scene_urdf,
    )
    - [**in use**]

- pick_and_place: 
    - create own env 
    - from .pick_and_place_articulation import ( 
        load_lab_wall,     load_table_4, )
    - original version without panda arm 

-  pick_and_place_panda & pick_and_place_panda_real & pick_and_place_panda_rl: 
    - create own env
    - from .pick_and_place_articulation import ( 
        load_lab_wall,     load_table_4, )
    - "_rl": define reward function
    - **Notice**: 
        **pick_and_place_panda**: original version [**in use**]
        **pick_and_place_panda_rl**: original version with rl
        **pick_and_place_panda_real**: real assets already be added, and added the calibrated camera (without rl) [**in use**]
        

- pick_and_place_bowl: 
    - create own env PickAndPlaceBowlEnv
    - from .pick_and_place_articulation import (
        load_lab_wall,
        load_storage_box,
        build_actor_ycb,
    )
    - [**in use**]

- drawer:
    - create own env PickAndPlaceEnv
    - from .drawer_articulation import (
        load_drawers,
        load_table_4,
    )
    - [**in use**]


### action explanation

#### step

- 7d(6+1) 
    - 当前输入的action是tool/base参考系下target tcp pose
    - 根据需要转换参考坐标系
        - tool系的参考是当前tcp pose在base系下的pose，cur.trans(action), 得到了base系下的target ee pose（采集的时候）
        - none， 得到的就是原系 （在验证的时候，因为已经是base系下的target tcp pose）
    - 然后将这个base系下的target ee pose转为world系下的target ee pose,  base.trans(..)
    - 再在world下解算target qpos，最后set 

#### compute (in collect)

- _desired_tcp_to_action

- 7d(6+1) 
    - 当前输入的是(某个fixed系)参考系下的target tcp pose
    - 先转为base系下的target tcp pose
    - 根据需要转换参考坐标系
        - tool系的参考是当前tcp pose在base系下的pose（在采集的时候）, cur.inv().trans(desired)
        - compute的时候不能是none（因为只有相对系下的pose作为action才有意义）
    - 然后将这个结果（tool系下的target pose）传出去作为expert action 来step（也用来采集作为数据中的action）
        - expert action就是收集数据时的action，也用来训练, 7d

#### predict

- 预测结果是base系下的delta pose（也可以理解为tool系的target pose），和base系下的pose_at_obs计算，得到base系下的target pose，作为step action


## /collect 

- collect_il: imitation data collection
- collect_pair: observation for random and canonical environment collection

### collect data format

imitation data:
- obs:
    - tcp_pose: 
        - 【actually in use】
        - ndarray, 7,  base-frame tcp pose [base_ee_pose]
        - self.init_base_pose.inv().transform(self._get_tcp_pose())
        - np.concatenate([tcp_pose.p, tcp_pose.q])
    - gripper_width: 
        - 【actually in use】
        - float32, robot.get_qpos()
        - self._get_gripper_width()
    - robot_joints: 
        - ndarray, 7 ,  robot's qpos [joint_pos]
        - self.robot.get_qpos()
    - privileged_obs: 
        - ndarray, 8, world-frame tcp pose [fixed_ee_pose]
        - world_tcp_pose = self._get_tcp_pose()
        - np.concatenate(
                [
                    world_tcp_pose.p,
                    world_tcp_pose.q,
                    [gripper_width],
                ]

- expert_action：
    - desired_grasp_pose: 
        - sapien.pose,  ee pose [fixed_ee_pose]
    - desired_gripper_width: float 
    - action: 
        - 【actually in use】
        - ndarray, 7,  [fixed/tool_ee_delta_pose], 注意收集的时候用的就是tool坐标系
        - np.concatenate(
                [
                    delta_pos,
                    delta_euler,
                    [gripper_width],
                ]

## /eval

- eval_imitation_diffusion: eval IL + diffusion 
- eval_replay: check the action of imitation data (.pkl)
- eval_il_server: eval IL + diffusion with ldm transfer