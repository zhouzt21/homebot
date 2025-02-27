import sys
from homebot_sapien.algorithm.imitation.dataset import export_video_from_demonstration

demo_file_name = sys.argv[1]
export_video_from_demonstration(demo_file_name, "traj.gif")
