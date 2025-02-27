
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