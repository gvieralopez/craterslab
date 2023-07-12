import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import Bounds, minimize

from craterslab.profiles import Profile
from craterslab.sensors import DepthMap

MIN_REQ_POINTS = 4


def fit_elipse(
    x: np.ndarray,
    y: np.ndarray,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    # Define the objective function for fitting the ellipse
    def objective(
        params: tuple[float, float, float, float, float], x: np.ndarray, y: np.ndarray
    ) -> float:
        a, b, cx, cy, theta = params
        ct = np.cos(theta)
        st = np.sin(theta)
        x_hat = ct * (x - cx) + st * (y - cy)
        y_hat = -st * (x - cx) + ct * (y - cy)
        ellipse = ((x_hat / a) ** 2 + (y_hat / b) ** 2) - 1
        return float(np.sum(ellipse**2))

    # Compute the mean and standard deviation of the data
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)

    # Define the initial guess for the ellipse parameters based on the mean and standard deviation of the data
    params0 = [x_std, y_std, x_mean, y_mean, 0]

    # Define the bounds for the ellipse
    lower_bound = [xrange[0], yrange[0], xrange[0], yrange[0], 0]
    upper_bound = [xrange[1], yrange[1], xrange[1], yrange[1], 2 * np.pi]
    bounds = Bounds(np.array(lower_bound), np.array(upper_bound))

    # Fit the ellipse to the points using the objective function and the initial guess
    result = minimize(objective, params0, args=(x, y), bounds=bounds)
    a, b, cx, cy, theta = result.x

    if a < b:
        a, b = b, a
        theta += np.pi/2
        theta = theta % np.pi

    # Extract the fitted ellipse parameters from the optimization result, and create the `Ellipse` patch using these parameters
    return a, b, cx, cy, theta


class EllipticalModel:
    """
    Computes an ellipse that fits the crater rims on a depth map
    """

    def __init__(self, depth_map: DepthMap, points: int) -> None:
        if not points % 2 == 0:
            raise ValueError("The value of points needs to be even")
        if points < MIN_REQ_POINTS:
            raise ValueError(
                f"The value of points needs to be greater than {MIN_REQ_POINTS}"
            )

        self.dm = depth_map
        self.points = points
        self._compute_landmark_points(points)
        if self._validate_landmark_points():
            self._compute_ellipse()

    def _compute_landmark_points(self, points: int) -> None:
        bounds = self._compute_bounds(points)
        self.landmarks = self._compute_landmarks(bounds)

    def _compute_bounds(
        self, points: int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        center_y, center_x = self.dm.y_count // 2, self.dm.x_count // 2
        angle_diff = 2 * np.pi / points
        return [
            self._compute_profile_bounds(i * angle_diff, center_x, center_y)
            for i in range(math.ceil(points / 2))
        ]

    def _compute_profile_bounds(
        self, angle: float, center_x: int, center_y: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        max_radius = self._compute_max_radius(angle, center_x, center_y)
        x0 = int(round(center_x + max_radius * np.cos(angle)))
        y0 = int(round(center_y + max_radius * np.sin(angle)))
        xf = int(round(center_x + max_radius * np.cos(angle + np.pi)))
        yf = int(round(center_y + max_radius * np.sin(angle + np.pi)))
        return (x0, y0), (xf, yf)

    def _compute_max_radius(self, angle: float, center_x: int, center_y: int) -> float:
        cos_angle = float(np.abs(np.cos(angle)))
        sin_angle = float(np.abs(np.sin(angle)))
        if cos_angle == 0:
            return center_y / sin_angle
        elif sin_angle == 0:
            return center_x / cos_angle
        return min(center_y / sin_angle, center_x / cos_angle)

    def _compute_landmarks(
        self, bounds: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> list[tuple[float, float]]:
        landmarks = []
        for bound in bounds:
            landmarks += self._compute_complementary_landmarks(bound)             
        return landmarks

    def _compute_complementary_landmarks(
        self, bound: tuple[tuple[int, int], tuple[int, int]]
    ) -> list[tuple[float, float]]:
        p = Profile(self.dm, bound[0], bound[1])
        # TODO: Improve this with a better way to get the extreme values
        p.compute_extrema()
        lm = []
        for i in [p.t1, p.t2]:
            if i == -1:
                logging.error("Cannot compute a landmark")
                continue
            x, y, z = p._index2xyz(i)
            lm.append((x, y))
        return lm


    def _validate_landmark_points(self) -> bool:
        if len(self.landmarks) < self.points:
            logging.warning(
                f"Only {len(self.landmarks)}/{self.points} landmarks were computed"
            )
            if len(self.landmarks) < MIN_REQ_POINTS:
                logging.error(
                    "Not sufficient landmarks for fitting the elliptical model"
                )
                return False
            else:
                logging.warning("Trying to fit the elliptical model anyways")
        return True

    def _compute_ellipse(self) -> None:
        x, y = map(np.array, zip(*self.landmarks))
        # x = np.array([p[0] for p in self.landmarks])
        # y = np.array([p[1] for p in self.landmarks])
        xrange, yrange = (0, self.dm.x_count), (0, self.dm.y_count)
        self.a, self.b, self.cx, self.cy, self.theta = fit_elipse(x, y, xrange, yrange)

    def ellipse_patch(self, scale: float = 1.0, color: str | None = None) -> Ellipse:
        cx, cy = self.cx, self.cy
        da, db = 2 * self.a, 2 * self.b
        theta_degree = self.theta * 180 / np.pi
        if scale != 1.0:
            cx, cy, da, db = (scale * i for i in [cx, cy, da, db])
        return Ellipse((cx, cy), da, db, theta_degree, fill=False, color=color)

    def max_profile(self) -> Profile:
        bound = self._compute_profile_bounds(self.theta, self.cx, self.cy)
        return Profile(self.dm, bound[0], bound[1])
