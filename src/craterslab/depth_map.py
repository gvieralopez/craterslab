import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from craterslab.images import compute_bounding_box, crop_img

VALID_SCALES = ["mm", "cm", "dm", "m", "km"]
AUTO_CROP_THRESHOLD = 1.5
AUTO_CROP_PADDING = 10


@dataclass
class SensorResolution:
    x: float
    y: float
    z: float
    scale: str = "mm"

    def __post_init__(self) -> None:
        if self.scale not in VALID_SCALES:
            raise ValueError("Unknown scale factor")

    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


class DepthMap:
    def __init__(self, dmap: np.ndarray, resolution: SensorResolution):
        self.map = dmap
        self.sensor = resolution
        self.x_res, self.y_res, self.z_res = resolution.xyz()

    def __sub__(self, other: "DepthMap") -> "DepthMap":
        # Validate the objects compatibility
        if not isinstance(other, DepthMap):
            raise TypeError("Cannot subtract non-DepthMap object from DepthMap")
        if self.map.shape != other.map.shape:
            raise ValueError("DepthMaps must have the same shape to subtract them")
        if self.sensor != other.sensor:
            raise ValueError("DepthMaps must have the same resolution to subtract them")

        # Subtract the data arrays elementwise
        new_data = self.map - other.map

        # Create a new DepthMap instance with the subtracted data
        return DepthMap(new_data, self.sensor)

    def crop(self, bounding_box: tuple[int, int, int, int], padding: int = 0) -> None:
        self.map = crop_img(self.map, *bounding_box, gap=padding)

    def crop_borders(self, ratio: float) -> None:
        if not 0 < ratio < 1:
            raise ValueError("The ratio needs to be between 0 and 1")
        x0, y0 = self.x_count * ratio / 2, self.y_count * ratio / 2
        left, top, width, height = map(int, [x0, y0, self.x_count - 2 * x0, self.y_count - 2 * y0])
        self.crop((left, top, width, height))

    def auto_crop(
        self, threshold: float = AUTO_CROP_THRESHOLD, padding: int = AUTO_CROP_PADDING
    ) -> None:
        bb = compute_bounding_box(self.map, threshold=threshold)
        self.crop(bb, padding)

    @property
    def heights(self) -> np.ndarray:
        return self.map * self.z_res

    @property
    def x_count(self) -> int:
        return int(self.map.shape[1])

    @property
    def y_count(self) -> int:
        return int(self.map.shape[0])

    def to_arrays(self) -> dict[str, Any]:
        return {
            "depthmap": self.map,
            "xres": self.sensor.x,
            "yres": self.sensor.y,
            "zres": self.sensor.z,
            "scale": self.sensor.scale,
        }

    def save(self, path: pathlib.Path | str) -> None:
        np.savez(path, **self.to_arrays())

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "DepthMap":
        with np.load(path) as data:
            sensor = SensorResolution(data["xres"], data["yres"], data["zres"], data["scale"])
            dm = data["depthmap"]
            return DepthMap(dm, sensor)
