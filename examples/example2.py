# Craterslab Example Script No. 2:
# Load a single depth map from file, compute an elliptical model and plot
# the depth map together with the model and its max profile

from craterslab.sensors import DepthMap
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.ellipse import EllipticalModel


depth_map = DepthMap.load('examples/data/fluidized_1.npz')
depth_map.auto_crop()

em = EllipticalModel(depth_map, 20)
p = em.max_profile()

plot_3D(depth_map, profile=p, ellipse=em, preview_scale=(1, 1, 4))
plot_2D(depth_map, profile=p, ellipse=em)
plot_profile(p, block=True)
