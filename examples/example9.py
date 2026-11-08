# Craterslab Example Script No. 9:
# Export the observables of several surfaces and a profile to CSV files

from craterslab import DepthMap, Surface, SurfaceType, write_observables_csv, write_profile_csv

surfaces = {}
for index in range(1, 6):
    depth_map = DepthMap.load(f"examples/data/fluidized_{index}.npz")
    depth_map.auto_crop()
    surfaces[f"fluidized_{index}"] = Surface.from_depth_map(depth_map, SurfaceType.SIMPLE_CRATER)

write_observables_csv("examples/data/observables.csv", surfaces)
write_profile_csv("examples/data/profile.csv", surfaces["fluidized_1"].max_profile)
