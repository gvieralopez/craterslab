import pathlib
from abc import classmethod, staticmethod
from dataclasses import dataclass

import numpy as np
import scipy.io

VALID_SCALES = ["mm", "cm", "dm", "m", "km"]


@dataclass
class SensorResolution:
    x: float
    y: float
    z: float
    scale: str = "mm"

    def __post_init__(self):
        if not self.scale in VALID_SCALES:
            raise ValueError("Unknown scale factor")


DEFAULT_RESOLUTION = SensorResolution(1.0, 1.0, 1.0)


class DepthMap:
    def __init__(self, dmap: np.ndarray, resolution: SensorResolution):
        self.map = dmap
        self.resolution = resolution
        self.x_res, self.y_res, self.z_res = resolution

    def __sub__(self, other) -> "DepthMap":
        # Validate the objects compatibility
        if not isinstance(other, DepthMap):
            raise TypeError("Cannot subtract non-DepthMap object from DepthMap")
        if self.map.shape != other.map.shape:
            raise ValueError("DepthMaps must have the same shape to subtract them")
        if self.resolution != other.resolution:
            raise ValueError("DepthMaps must have the same resolution to subtract them")

        # Subtract the data arrays elementwise
        new_data = self.map - other.map

        # Create a new DepthMap instance with the subtracted data
        return DepthMap(new_data, self.resolution)

    @classmethod
    def from_mat_file(
        cls,
        file_name: str,
        data_folder: str = "data",
        max_samples: int = -1,
        average: bool = True,
        resolution: SensorResolution = DEFAULT_RESOLUTION,
    ) -> "DepthMap":

        # Load depth map from file
        content = cls._from_mat_file(file_name, data_folder)

        # Reduce the amount of raw data used
        if max_samples != -1:
            content = content[:, :, :, :max_samples]

        # Average the last dimension to obtain an image-like numpy array
        if average:
            content = np.mean(content, axis=-1).squeeze()

        return cls(content, resolution)

    @classmethod
    def from_image_file(cls, image) -> "DepthMap":
        # Compute depth map from image
        raise NotImplementedError

    @classmethod
    def from_kinect_sensor(cls) -> "DepthMap":
        # Retrieve depth map from Kinect sensor
        raise NotImplementedError

    @staticmethod
    def _from_mat_file(
        file_name: str, data_folder: str, variable_name: str | None
    ) -> np.ndarray:

        # Load the .mat file
        file_path = pathlib.Path(data_folder, file_name)
        mat_contents = scipy.io.loadmat(file_path)

        # Access the desired variable from the .mat file
        variable_name = variable_name if variable_name else file_name[:-4]
        return mat_contents[variable_name]
