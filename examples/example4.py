# Craterslab Example Script No. 4:
# Compute the diameters from the surfaces found across different depth maps

from craterslab.craters import Surface
from craterslab.sensors import DepthMap


diameters = []
for index in range(1, 37):
    print(f'Analyzing file {index}')
    depth_map = DepthMap.load(f'examples/data/fluidized_{index}.npz')
    depth_map.auto_crop()
    s = Surface(depth_map)
    diameters.append(s.observables["D"].value)

print(diameters)
