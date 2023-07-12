import math

import numpy as np
import scipy

from craterslab.sensors import DepthMap


class Profile:
    """
    Computes the profile through a segment of a depth map:
    """

    def __init__(
        self,
        depth_map: DepthMap,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        total_points: int = 1000,
    ):
        self.dm = depth_map
        self.x0, self.y0 = start_point
        self.xf, self.yf = end_point
        self.total_points = total_points

        assert depth_map.x_res == depth_map.y_res
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
        self._h = zi.flatten()

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
    def x_bounds_px(self) -> tuple[int, int]:
        return self.x0, self.xf

    @property
    def y_bounds_px(self) -> tuple[int, int]:
        return self.y0, self.yf

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (self.x0 * self.s_res, self.xf * self.s_res)

    @property
    def y_bounds(self) -> tuple[float, float]:
        return (self.y0 * self.s_res, self.yf * self.s_res)

    @property
    def z_bounds(self) -> tuple[float, float]:
        return (min(self.h) * self.h_res, max(self.h) * self.h_res)

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

    # TODO: Temporal methods:
    def compute_extrema(self):
        dhds = np.gradient(self._h, self._s)
        extrema_indices = np.where(np.diff(np.sign(dhds)))[0]
        sorted_indices = extrema_indices[np.argsort(self.h[extrema_indices])]

        self.b1 = next(filter(lambda i: i <= len(self) // 2, sorted_indices), -1)
        self.b2 = next(filter(lambda i: i > len(self) // 2, sorted_indices), -1)

        indices = sorted_indices[::-1]
        third = len(self) // 3
        self.t1 = next(filter(lambda i: i <= third, indices), -1)
        self.tc = next(filter(lambda i: third < i <= 2 * third, indices), -1)
        self.t2 = next(filter(lambda i: i > 2 * third, indices), -1)

    # @property
    # def walls(self):
    #     """
    #     Provide the points that bound each crater wall in the profile.
    #     Result is returned as: [(x11, x12), (z11, z12)], [(x21, x22), (z21, z22)].
    #     So, first the bounds for the left wall and then the bounds for the right
    #     wall.
    #     """
    #     left_wall = self._compute_wall_bounds(1)
    #     right_wall = self._compute_wall_bounds(2)
    #     return left_wall, right_wall

    # def _compute_slope(self, i: int, j: int):
    #     """
    #     Compute the slope of the line that better fits the profile from i to j
    #     """
    #     if None in (i, j):
    #         raise ValueError("Invalid indices for slope calculation.")
    #     i, j = (i, j) if i < j else (j, i)
    #     x_vals, y_vals = self.s[i:j], self.h[i:j]
    #     if x_vals.size > 1:
    #         try:
    #             p1 = np.polyfit(x_vals, y_vals, 1)
    #             return np.poly1d(p1)
    #         except SystemError:
    #             logging.error("Error computing slopes")

    # def _compute_slopes(self):
    #     """
    #     Compute the slope of crater walls
    #     """
    #     self.slope1 = self._compute_slope(self.t1, self.b1)
    #     self.slope2 = self._compute_slope(self.b2, self.t2)

    # def _compute_wall_bounds(self, wall: int = 1):
    #     """
    #     Compute the bounds of a given wall given the id (1 or 2) to refer to
    #     the left or right wall respectively.
    #     """
    #     assert wall in (1, 2)
    #     if wall == 1:
    #         return self._single_wall_bound(self.t1, self.b1, self.slope1)
    #     return self._single_wall_bound(self.t2, self.b2, self.slope2)

    # def _single_wall_bound(self, i: int, j: int, model: Callable):
    #     """
    #     Return the wall bounds given the indexes of the bounds (i, j) and the
    #     linear function that interpolates them.
    #     """
    #     x = [self.s[i], self.s[j]]
    #     if model:
    #         z = [model(v) for v in x]
    #     else:
    #         z = x
    #         logging.error("Slopes are invalid")
    #     return x, z
