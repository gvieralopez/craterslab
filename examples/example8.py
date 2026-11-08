# Craterslab Example Script No. 8:
# Save a classified surface and load it back without classifying it again

from craterslab.classification import SurfaceType
from craterslab.craters import Surface
from craterslab.depth_map import DepthMap

depth_map = DepthMap.load("examples/data/fluidized_1.npz")
depth_map.auto_crop()

surface = Surface.from_depth_map(depth_map, SurfaceType.SIMPLE_CRATER)
surface.save("examples/data/my_crater.npz")

loaded = Surface.load("examples/data/my_crater.npz")
print(loaded)
