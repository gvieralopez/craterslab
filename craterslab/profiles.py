import math
import logging

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

        indices = sorted_indices[::-1]
        third = len(self) // 3
        self.t1 = next(filter(lambda i: i <= third, indices), -1)
        self.tc = next(filter(lambda i: third < i <= 2 * third, indices), -1)
        self.t2 = next(filter(lambda i: i > 2 * third, indices), -1)

        self.b1 = next(
            filter(lambda i: i <= len(self) // 2 and i > self.t1, sorted_indices), -1
        )
        self.b2 = next(
            filter(lambda i: i > len(self) // 2 and i < self.t2, sorted_indices), -1
        )

        # If b1 or b2 have not been found
        if self.b1 == -1 or self.b2 == -1:
            # Check if only one is missing
            if self.b1 != self.b2:
                self.b1 = self.b2 if self.b1 == -1 else self.b1
                self.b2 = self.b1 if self.b2 == -1 else self.b2

    def slopes(
        self, t1: int = -1, b1: int = -1, t2: int = -1, b2: int = -1
    ) -> tuple[float, float]:
        self.set_landmark_indices(t1, b1, t2, b2)
        self.l1, self.l2 = self._compute_linear_fit()
        return self.l1[0], self.l2[0]

    def set_landmark_indices(
        self, t1: int = -1, b1: int = -1, t2: int = -1, b2: int = -1
    ):
        self.compute_extrema()
        self.t1 = t1 if t1 != -1 else self.t1
        self.t2 = t2 if t2 != -1 else self.t2
        self.b1 = b1 if b1 != -1 else self.b1
        self.b2 = b2 if b2 != -1 else self.b2

    def _compute_linear_fit(self) -> tuple[np.array, np.array]:
        """
        Compute the lines that better fits the crater walls
        """
        return self._compute_line(self.t1, self.b1), self._compute_line(
            self.b2, self.t2
        )

    def _compute_line(self, i: int, j: int) -> np.array:
        """
        Compute the line that better fits the profile from i to j
        """
        if None in (i, j):
            raise ValueError("Invalid indices for slope calculation.")
        i, j = (i, j) if i < j else (j, i)
        x_vals, y_vals = self.s[i:j], self.h[i:j]
        if x_vals.size > 1:
            try:
                return np.polyfit(x_vals, y_vals, 1)
            except SystemError:
                logging.error("Error computing slopes")
