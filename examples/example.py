from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.craters import Surface

# Define sensor resolution
KINECT_RESOLUTION = SensorResolution(2.8025, 2.8025, 1.0)

# Define data sources
d0 = DepthMap.from_mat_file("plano_aFig4.mat", resolution=KINECT_RESOLUTION)
df = DepthMap.from_mat_file("crater_aFig4.mat", resolution=KINECT_RESOLUTION)

# Compute the difference between the surface before and after the impact
depth_map = d0 - df
depth_map.auto_crop()

s = Surface(depth_map)

plot_3D(depth_map, profile=s.max_profile, ellipse=s.em, preview_scale=(1, 1, 4))
plot_2D(depth_map, profile=s.max_profile, ellipse=s.em)
plot_profile(s.max_profile, block=True)

print(s)
