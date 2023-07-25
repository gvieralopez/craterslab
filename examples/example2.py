from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.ellipse import EllipticalModel

# Define sensor resolution
KINECT_RESOLUTION = SensorResolution(2.8025, 2.8025, 1.0)


for i in range(1, 30):
    print(f"Analizing sample {i}")

    # Define data sources
    d0 = DepthMap.from_mat_file(
        f"planoexp{i}.mat",
        data_folder="data/Fluized_sand",
        resolution=KINECT_RESOLUTION,
    )
    df = DepthMap.from_mat_file(
        f"craterexp{i}.mat",
        data_folder="data/Fluized_sand",
        resolution=KINECT_RESOLUTION,
    )

    # Compute the difference between the surface before and after the impact
    depth_map = d0 - df
    depth_map.auto_crop()

    em = EllipticalModel(depth_map, 20)
    p = em.max_profile()

    plot_3D(depth_map, profile=p, ellipse=em, preview_scale=(1, 1, 4))
    plot_2D(depth_map, profile=p, ellipse=em)
    plot_profile(p, block=True)