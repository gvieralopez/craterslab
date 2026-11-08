# Craterslab Example Script No. 5:
# Customizing craterslab plots

from craterslab.classification import SurfaceType
from craterslab.craters import Surface
from craterslab.depth_map import DepthMap
from craterslab.ellipse import EllipseVisualConfig
from craterslab.visuals import plot_3D

depth_map = DepthMap.load("examples/data/fluidized_1.npz")
depth_map.auto_crop()

s = Surface.from_depth_map(depth_map, SurfaceType.SIMPLE_CRATER)

ellipse_config = EllipseVisualConfig(
    color="blue", fill=True, z_val=s.observables["mean_h_rim"].value, alpha=0.5
)

plot_3D(depth_map, ellipse=s.em, preview_scale=(1, 1, 4), ellipse_config=ellipse_config, block=True)
