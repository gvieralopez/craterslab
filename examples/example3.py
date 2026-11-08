# Craterslab Example Script No. 3:
# Load a cloud point from an xyz file containing the data from the King crater.
# The crater is then analyzed using craterslab functionalities.

from craterslab.cnn import CNNClassifier
from craterslab.craters import Surface
from craterslab.depth_map import SensorResolution
from craterslab.readers import read_xyz
from craterslab.visuals import plot_2D, plot_3D, plot_profile

# Define sensor resolution
data_resolution = SensorResolution(235.65, 235.65, 1.0, "m")

# Define data sources
depth_map = read_xyz("examples/data/king.xyz", data_resolution, z_shift=-1445)
depth_map.crop_borders(ratio=0.25)

# Create a surface object from the depth map and classify it
s = Surface.from_depth_map(depth_map)
s.classify(CNNClassifier())

# Produce the plots
plot_3D(depth_map, profile=s.max_profile, ellipse=s.em, preview_scale=(1, 1, 5))
plot_2D(depth_map, profile=s.max_profile, ellipse=s.em)
plot_profile(s.max_profile, block=True)

# Output the observables computed for the crater
print(s)
