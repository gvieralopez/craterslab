import math

import numpy as np
import scipy

from craterslab.depth_map import DepthMap


class Profile:
    """
    Computes the profile through a segment of a depth map:
    """

    def __init__(
        self,
        depth_map: DepthMap,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        total_points: int = 1000,
    ):
        self.dm = depth_map
        self.x0, self.y0 = start_point
        self.xf, self.yf = end_point
        self.total_points = total_points

        if depth_map.x_res != depth_map.y_res:
            raise ValueError("Profiles need a depth map with equal x and y resolution")
        self.s_res, self.h_res = depth_map.x_res, depth_map.z_res
        self._compute_height()
        self._compute_displacement()

    def _compute_height(self) -> None:
        """
        Compute height (h) for each profile point according to the depth-map img
        """
        x = np.linspace(self.x0, self.xf, self.total_points)
        y = np.linspace(self.y0, self.yf, self.total_points)
        zi = scipy.ndimage.map_coordinates(self.dm.map, np.vstack((y, x)))
        self._h: np.ndarray = zi.flatten()

    def _compute_displacement(self) -> None:
        """
        Compute the displacement (s) on each profile point
        """
        dx, dy = self.xf - self.x0, self.yf - self.y0
        max_displacement = np.sqrt(dx**2 + dy**2)
        self._s = np.linspace(0, max_displacement, self.total_points)

    def __len__(self) -> int:
        return len(self._h)

    @property
    def x_bounds_px(self) -> tuple[float, float]:
        return self.x0, self.xf

    @property
    def y_bounds_px(self) -> tuple[float, float]:
        return self.y0, self.yf

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (self.x0 * self.s_res, self.xf * self.s_res)

    @property
    def y_bounds(self) -> tuple[float, float]:
        return (self.y0 * self.s_res, self.yf * self.s_res)

    @property
    def z_bounds(self) -> tuple[float, float]:
        return (min(self.h), max(self.h))

    @property
    def ang(self) -> float:
        """
        Provide the angle of the profile
        """
        return math.atan2(self.yf - self.y0, self.xf - self.x0)

    @property
    def h(self) -> np.ndarray:
        """
        Provide the height profile of the crater on the original scale
        """
        return self.h_res * self._h

    @property
    def s(self) -> np.ndarray:
        """
        Provide the displacement array on which the height profile was computed
        on the original scale
        """
        return self.s_res * self._s

    def _index2xyz(self, index: int) -> tuple[float, float, float]:
        si, zi = self._s[index], self._h[index]
        xi = self.x0 + si * math.cos(self.ang)
        yi = self.y0 + si * math.sin(self.ang)
        return xi, yi, zi

    def index2xyz_pix(self, index: int) -> tuple[int, int, int]:
        xi, yi, zi = self._index2xyz(index)
        return int(xi), int(yi), int(zi)

    def index2xyz(self, index: int) -> tuple[float, float, float]:
        xi, yi, zi = self._index2xyz(index)
        return xi * self.s_res, yi * self.s_res, zi * self.h_res
