# Craterslab Example Script No. 5:
# Customizing craterslab plots

from craterslab.ellipse import EllipseVisualConfig
from craterslab.sensors import DepthMap
from craterslab.visuals import plot_3D
from craterslab.craters import Surface


depth_map = DepthMap.load('examples/data/fluidized_1.npz')
depth_map.auto_crop()

s = Surface(depth_map)

ellipse_config = EllipseVisualConfig(
    color="blue", fill=True, z_val=s.observables["mean_h_rim"].value, alpha=0.5
)

plot_3D(
    depth_map,
    ellipse=s.em,
    preview_scale=(1, 1, 4),
    ellipse_config=ellipse_config,
    block=True,
)
