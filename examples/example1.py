# Craterslab Example Script No. 1:
# Load a single depth map from file and plot it

from craterslab.sensors import DepthMap
from craterslab.visuals import plot_2D, plot_3D

depth_map = DepthMap.load('examples/data/fluidized_1.npz')
depth_map.auto_crop()

plot_3D(depth_map, preview_scale=(1, 1, 4))
plot_2D(depth_map, block=True)