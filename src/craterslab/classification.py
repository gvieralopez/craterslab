from enum import Enum
from typing import Protocol

from craterslab.depth_map import DepthMap


class SurfaceType(Enum):
    UNKNOWN = 0
    SIMPLE_CRATER = 1
    COMPLEX_CRATER = 2
    SAND_MOUND = 3

    def __str__(self) -> str:
        return self.name.replace("_", " ").capitalize()


class Classifier(Protocol):
    def classify(self, depth_map: DepthMap) -> SurfaceType: ...
