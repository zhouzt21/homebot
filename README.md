# homebot_sapien
The simulation environment for RL training.

## Installation
From docker:
```
cd docker
docker build -t sapien -f Dockerfile .
```

## Usage
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
