# Craterslab Example Script No. 4:
# Compute the diameters from the surfaces found across different depth maps

from craterslab import DepthMap, Surface
from craterslab.cnn import CNNClassifier

classifier = CNNClassifier()
diameters = []
for index in range(1, 37):
    print(f"Analyzing file {index}")
    depth_map = DepthMap.load(f"examples/data/fluidized_{index}.npz")
    depth_map.auto_crop()
    s = Surface.from_depth_map(depth_map)
    s.classify(classifier)
    d = s.observables["D"].value if "D" in s.observables else -1
    diameters.append(d)

print(diameters)
