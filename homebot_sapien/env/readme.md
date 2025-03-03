
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
    - eval_imitation() 感觉是训完了之后来验证训的效果（验证diffusion）

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
