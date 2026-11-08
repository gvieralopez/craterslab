from pathlib import Path

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter

from craterslab.depth_map import DepthMap, SensorResolution


def read_mat(
    path: Path | str,
    resolution: SensorResolution,
    max_samples: int | None = None,
    average: bool = True,
    variable_name: str | None = None,
) -> DepthMap:
    file_path = Path(path)
    content = scipy.io.loadmat(file_path)[variable_name or file_path.stem]
    if max_samples is not None:
        content = content[:, :, :, :max_samples]
    if average:
        content = np.mean(content, axis=-1).squeeze()
    return DepthMap(content, resolution)


def read_xyz(
    path: Path | str,
    resolution: SensorResolution,
    z_shift: float = 0,
    rescaled_with: tuple[float, float, float] | None = None,
) -> DepthMap:
    points = np.loadtxt(path, skiprows=2)
    points[:, 2] += z_shift
    if rescaled_with is not None:
        points = _rescale(points, rescaled_with)
    return DepthMap(_point_cloud_to_depth_map(points, resolution), resolution)


def _rescale(points: np.ndarray, scale: tuple[float, float, float]) -> np.ndarray:
    for axis in range(3):
        points[:, axis] *= scale[axis]
    return points


def _point_cloud_to_depth_map(points: np.ndarray, resolution: SensorResolution) -> np.ndarray:
    x, y, z = points.T
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
    for xi, yi, zi in zip(_x, _y, z, strict=True):
        matrix[yi, xi] += zi
        point_count[yi, xi] += 1

    # Divide each matrix element by the number of points that were mapped to it
    result = np.zeros_like(matrix)
    np.divide(matrix, point_count, out=result, where=point_count != 0)

    smoothed: np.ndarray = gaussian_filter(result, sigma=2)
    return smoothed
