from craterslab.ellipse import EllipseVisualConfig
from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_3D
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
