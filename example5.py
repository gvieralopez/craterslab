from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.craters import Surface

# Define sensor resolution
data_resolution = SensorResolution(235.65, 235.65, 1.0, "m")

# Define data sources
depth_map = DepthMap.from_xyz_file(
    "king.xyz", resolution=data_resolution, rescaled_with=(1, 1, 1)
)
depth_map.crop_borders(ratio=0.25)

s = Surface(depth_map)

plot_3D(depth_map, profile=s.max_profile, ellipse=s.em, preview_scale=(1, 1, 5))
plot_2D(depth_map, profile=s.max_profile, ellipse=s.em)
plot_profile(s.max_profile, block=True)

print(s)
