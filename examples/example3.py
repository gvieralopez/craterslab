# Craterslab Example Script No. 3:
# Load a cloud point from an xyz file containing the data from the King crater.
# The crater is then analyzed using craterslab functionalities.

from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.craters import Surface

# Define sensor resolution
data_resolution = SensorResolution(235.65, 235.65, 1.0, "m")

# Define data sources
depth_map = DepthMap.from_xyz_file(
    "king.xyz", data_folder="examples/data/", resolution=data_resolution
)
depth_map.crop_borders(ratio=0.25)

# Create a surface object from the depth map
s = Surface(depth_map)

# Produce the plots
plot_3D(depth_map, profile=s.max_profile, ellipse=s.em, preview_scale=(1, 1, 5))
plot_2D(depth_map, profile=s.max_profile, ellipse=s.em)
plot_profile(s.max_profile, block=True)

# Output the observables computed for the crater
print(s)
