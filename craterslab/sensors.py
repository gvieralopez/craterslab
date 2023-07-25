import pathlib
from dataclasses import dataclass

import numpy as np
import scipy.io
from scipy.ndimage.filters import gaussian_filter

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
        if not self.scale in VALID_SCALES:
            raise ValueError("Unknown scale factor")

    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


DEFAULT_RESOLUTION = SensorResolution(1.0, 1.0, 1.0)


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

    def crop_borders(self, ratio: float):
        assert ratio > 0 and ratio < 1
        x0, y0 = self.x_count * ratio / 2, self.y_count * ratio / 2
        bb = map(int, [x0, y0, self.x_count - 2 * x0, self.y_count - 2 * y0])
        self.crop(tuple(bb))

    def auto_crop(
        self, threshold: float = AUTO_CROP_THRESHOLD, padding: int = AUTO_CROP_PADDING
    ) -> None:
        bb = compute_bounding_box(self.map, threshold=threshold)
        self.crop(bb, padding)

    @property
    def x_count(self) -> int:
        return self.map.shape[1]

    @property
    def y_count(self) -> int:
        return self.map.shape[0]

    @classmethod
    def from_mat_file(
        cls,
        file_name: str,
        data_folder: str = "data",
        max_samples: int = -1,
        average: bool = True,
        variable_name: str | None = None,
        resolution: SensorResolution = DEFAULT_RESOLUTION,
    ) -> "DepthMap":
        # Load depth map from file
        content = cls._from_mat_file(file_name, data_folder, variable_name)

        # Reduce the amount of raw data used
        if max_samples != -1:
            content = content[:, :, :, :max_samples]

        # Average the last dimension to obtain an image-like numpy array
        if average:
            content = np.mean(content, axis=-1).squeeze()

        return cls(content, resolution)

    @classmethod
    def from_xyz_file(
        cls,
        file_name: str,
        data_folder: str = "data",
        rescaled_with: tuple[float, float, float] | None = None,
        resolution: SensorResolution = DEFAULT_RESOLUTION,
    ) -> "DepthMap":
        # Load cloud point from file
        content = cls._from_xyz_file(file_name, data_folder)

        if rescaled_with is not None:
            content = cls._rescale(content, rescaled_with)

        matrix = cls._point_cloud_to_depth_map(content, resolution)

        return cls(matrix, resolution)

    @classmethod
    def from_image_file(cls, image: np.ndarray) -> "DepthMap":
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

    @staticmethod
    def _from_xyz_file(file_name: str, data_folder: str) -> np.ndarray:
        file_path = pathlib.Path(data_folder, file_name)
        return np.loadtxt(file_path, skiprows=2)

    @staticmethod
    def _rescale(array, scale):
        for i in range(3):
            if scale[i] != 1:
                array *= scale[i]
        return array

    @staticmethod
    def _point_cloud_to_depth_map(point_cloud, resolution: SensorResolution):
        x, y, z = point_cloud.T
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # Compute target depth map shape
        cols = int((x_max - x_min) / resolution.x)
        rows = int((y_max - y_min) / resolution.y)
        matrix = np.zeros((rows, cols), dtype=np.float64)
        point_count = np.zeros((rows, cols), dtype=np.int32)

        # Scale the x and y coordinates to fit the matrix
        _x = np.clip(np.round((x - x_min) / resolution.x).astype(int), 0, cols - 1)
        _y = np.clip(np.round((y - y_min) / resolution.y).astype(int), 0, rows - 1)

        # Assign the z values to the corresponding matrix elements
        for xi, yi, zi in zip(_x, _y, z):
            matrix[yi, xi] += zi
            point_count[yi, xi] += 1

        # Divide each matrix element by the number of points that were mapped to it
        result = np.zeros_like(matrix)
        np.divide(matrix, point_count, out=result, where=point_count != 0)

        return gaussian_filter(result, sigma=2)

    def save(self, path: pathlib.Path | str):
        np.savez(
            path,
            depthmap=self.map,
            xres=self.sensor.x,
            yres=self.sensor.y,
            zres=self.sensor.z,
            scale=self.sensor.scale,
        )

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "DepthMap":
        with np.load(path) as data:
            sensor = SensorResolution(
                data["xres"], data["yres"], data["zres"], data["scale"]
            )
            dm = data["depthmap"]
            return DepthMap(dm, sensor)
