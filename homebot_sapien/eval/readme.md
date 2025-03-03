
## articulation define

door_articulation
drawer_articulation
pick_and_place_articulation

## env define 

- open_door: 
    - create own env OpenDoor (only registered)
    - from .door_articulation import (
        load_lab_door,
        generate_rand_door_config,
        load_lab_wall,
        load_lab_scene_urdf,
    )

- pick_and_place: 
    - create own env PickAndPlaceEnv
    - from .pick_and_place_articulation import ( 
        load_lab_wall,     load_table_4, )

- pick_and_place_panda & pick_and_place_panda_rl: 
    - create own env
    - from .pick_and_place_articulation import ( 
        load_lab_wall,     load_table_4, )
    - "_rl": define reward function

- pick_and_place_bowl: 
    - create own env PickAndPlaceBowlEnv
    - from .pick_and_place_articulation import (
        load_lab_wall,
        load_storage_box,
        build_actor_ycb,
    )

- drawer:
    - create own env PickAndPlaceEnv
    - from .drawer_articulation import (
        load_drawers,
        load_table_4,
    )

## algos

- pick_and_place
    - collect_rand_and_cano_data(): sim2sim_3 这里只是保存了cano 和 rand 的image obs
    - collect_imitation_data(): cano_policy_2 这里采集了action state image所有数据
    - 这里开始训diffusion policy了
    - eval_imitation() 训完了之后来验证训的效果（验证diffusion）

- pick_and_place_panda
    - collect_rand_and_cano_data(): sim2sim_pd_1
    - collect_imitation_data(): rand_policy_pd_1"
    // rand开头的是无rl panda的il数据； sim2sim数字开头

- pick_and_place_panda_rl
    - collect_imitation_data(): cano_policy_pd_rl_1

- drawer 
    - collect_imitation_data(): cano_drawer_0919
    - collect_sim2sim_data(): sim2sim_drawer_0919

- pick_and_place_bowl
    <!-- - collect_rand_and_cano_data(): sim2sim_pd_1 -->
    - collect_imitation_data(): cano_bowl_1007
    - collect_sim2sim_data(): sim2sim_bowl_1004    

- eval_imitation_diffusion
    - pick and place evaluation

## action 

#### step

- 7d(6+1) 
    - 当前输入的action是tool/base参考系下target tcp pose
    - 根据需要转换参考坐标系
        - tool系的参考是当前tcp pose在base系下的pose，cur.trans(action), 得到了base系下的target ee pose（采集的时候）
        - none， 得到的就是原系 （在验证的时候，因为已经是base系下的target tcp pose）
    - 然后将这个base系下的target ee pose转为world系下的target ee pose,  base.trans(..)
    - 再在world下解算target qpos，最后set 

#### compute (in collect)
_desired_tcp_to_action

- 7d(6+1) 
    - 当前输入的是(某个fixed系)参考系下的target tcp pose
    - 先转为base系下的target tcp pose
    - 根据需要转换参考坐标系
        - tool系的参考是当前tcp pose在base系下的pose（在采集的时候）, cur.inv().trans(desired)
        - compute的时候不能是none（因为只有相对系下的pose作为action才有意义）
    - 然后将这个结果（tool系下的target pose）传出去作为expert action 来step（采集）
        - expert action就是收集数据时的action，也用来训练

#### predict

- 预测结果是base系下的delta pose（也可以理解为tool系的target pose），和base系下的pose_at_obs计算，得到base系下的target pose，作为step action


## collect data format

imitation data:、
- obs:
    - tcp_pose: 
        - 【】
        - ndarray, 7, 是相对base的tcp pose [base_ee_pose]
        - self.init_base_pose.inv().transform(self._get_tcp_pose())
        - np.concatenate([tcp_pose.p, tcp_pose.q])
    - gripper_width: 
        - 【】
        - float32, robot.get_qpos()得到的
        - self._get_gripper_width()
    - robot_joints: 
        - ndarray, 7 , 从robot的qpos得到 [joint_pos]
        - self.robot.get_qpos()
    - privileged_obs: 
        - ndarray, 8, 是世界坐标系的tcp pose [fixed_ee_pose]
        - world_tcp_pose = self._get_tcp_pose()
        - np.concatenate(
                [
                    world_tcp_pose.p,
                    world_tcp_pose.q,
                    [gripper_width],
                ]

仿真中专家给的位姿：
- desired_grasp_pose: (不用)
    - sapien.pose,  ee pose [fixed_ee_pose]
- desired_gripper_width: float (不用)
- action: 
    - 【】
    - ndarray, 7,  [fixed/tool_ee_delta_pose], 注意收集的时候用的就是tool坐标系
    - np.concatenate(
            [
                delta_pos,
                delta_euler,
                [gripper_width],
            ]