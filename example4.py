from craterslab.craters import Surface
from craterslab.sensors import DepthMap, SensorResolution

# Define sensor resolution
KINECT_RESOLUTION = SensorResolution(2.8025, 2.8025, 1.0)


diameters = []
for i in range(1, 38):
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

    s = Surface(depth_map)
    diameters.append(s.observables['D'].value)

print(diameters)

