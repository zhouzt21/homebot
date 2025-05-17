# homebot_sapien
The simulation environment for RL training.

## Installation
From docker:
```
cd docker
docker build -t sapien -f Dockerfile .
```

Local:

```
conda create -n homebot python=3.8 -y
conda activate homebot
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## Old Version Usage

Scripted policy: it will export a video at `test.mp4`.
```
python -m homebot_sapien.env.open_door
```

Collect demonstration with parallel environments from the expert policy:
```
python collect_expert_demo.py
```
The demonstrations are saved under `./demos/scripted_opendoor`.

To visualize a demo trajectory, run
```
python visualize_demonstration.py [path/to/traj-xxx.pkl]
```
The visualization is saved as `traj.gif` with `fps=6`.

Imitation learning using diffusion policy,
```
python -m homebot_sapien.algorithm.imitate_diffuse
```

## New Version Usage

### Environment 

- already in use:
  - pick_and_place_panda: PickAndPlaceEnv, object dataset includes egad dataset, ycb dataset
  - pick_and_place_panda_real: PickAndPlaceEnv, object dataset includes egad dataset, ycb dataset, some real object (3d-reconstruction)
  - pick_and_place_panda_side: PickAndPlaceEnv, object dataset includes only real side objects
  - open_door: OpenDoorEnv
  - drawer, ,microwave: PushAndPullEnv
  

### collect

- `homebot/homebot_sapien/collect`
  - collect il env data for diffusion policy
    - `collect_il_real.py`, `collect_il_push_pull.py`, `collect_il_open_door.py` is in use
    - need to change `num_seeds`, `save_dir` and `name`
    - check the env to see if it is the one to collect with
  
### eval 

- `homebot/homebot_sapien/eval/eval_imitation_diffusion`
  - need to change the path of `*.ckpt` and `norm_stats_1.pkl` 
    - `norm_stats_1.pkl` is in data folder, generate automaticly when trainning diffusion_policy
  - need to change `num_seeds`, `save_dir` 
  - check the env to see if it is the one to evaluate with
