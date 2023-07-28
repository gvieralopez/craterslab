# Craterslab Example Script No. 6:
# Fetching depth maps from kinect sensor

from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_3D

# Establish sensor resolution obtained expermentally
resolution = SensorResolution(2.8025, 2.8025, 1.0, 'mm')

# Ask the kinect sensor for a depth map
depth_map = DepthMap.from_kinect_sensor(resolution, average_on=500)

# Plot the resulting depth map
plot_3D(depth_map, block=True)

# Save the depth map
depth_map.save('depthmap_kinect.npz')
