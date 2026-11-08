import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import Bounds, minimize

from craterslab.profiles import Profile
from craterslab.sensors import DepthMap

logger = logging.getLogger(__name__)

MIN_REQ_POINTS = 4
RIM_TOLERANCE = 0.05

Point = tuple[float, float]


def fit_elipse(
    x: np.ndarray, y: np.ndarray, xrange: tuple[float, float], yrange: tuple[float, float]
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
        theta += np.pi / 2
        theta = theta % np.pi

    # Extract the fitted ellipse parameters from the optimization result, and create the `Ellipse` patch using these parameters
    return a, b, cx, cy, theta


@dataclass(frozen=True)
class EllipseFit:
    a: float
    b: float
    cx: float
    cy: float
    theta: float
    landmarks: list[Point]


@dataclass
class EllipseVisualConfig:
    color: str = "red"
    fill: bool = False
    z_val: float = 0
    alpha: float = 1.0


class EllipticalModel:
    """
    Computes an ellipse that fits the crater rims on a depth map
    """

    def __init__(
        self, depth_map: DepthMap, points: int, inner: bool = False, fit: EllipseFit | None = None
    ) -> None:
        if not points % 2 == 0:
            raise ValueError("The value of points needs to be even")
        if points < MIN_REQ_POINTS:
            raise ValueError(f"The value of points needs to be greater than {MIN_REQ_POINTS}")

        self.dm = depth_map
        self.points = points
        self.visual = EllipseVisualConfig(z_val=np.max(depth_map.map))
        if fit is not None:
            self.a, self.b, self.cx, self.cy, self.theta = fit.a, fit.b, fit.cx, fit.cy, fit.theta
            self.landmarks = fit.landmarks
            return
        self._compute_landmark_points(points, inner)
        if not self._validate_landmark_points():
            raise ValueError("Not sufficient landmarks for fitting the elliptical model")
        self._compute_ellipse()

    def _compute_landmark_points(self, points: int, inner: bool) -> None:
        bounds = self._compute_bounds(points)
        self.landmarks = self._compute_landmarks(bounds, inner)

    def _compute_bounds(self, points: int) -> list[Sequence[Point]]:
        center_y, center_x = self.dm.y_count // 2, self.dm.x_count // 2
        angle_diff = 2 * np.pi / points
        return [
            self._compute_profile_bounds(i * angle_diff, center_x, center_y)
            for i in range(math.ceil(points / 2))
        ]

    def _compute_profile_bounds(
        self, angle: float, center_x: float, center_y: float
    ) -> Sequence[Point]:
        # Calculate the slope and y-intercept of the line
        m = np.tan(angle)
        n = center_y - m * center_x

        # Define the bounds
        x_max, y_max = self.dm.x_count - 1, self.dm.y_count - 1

        # Calculate the x- and y-coordinates of the four possible intersection points
        x1, y1 = 0, n
        x2 = x_max
        y2 = m * x2 + n
        if m == 0:
            return (x1, y1), (x2, y2)
        y3 = 0
        x3 = (y3 - n) / m
        y4 = y_max
        x4 = (y4 - n) / m

        # Find the two intersection points that lie on the border of the box
        points = []
        if (0 <= x1 <= x_max) and (0 <= y1 <= y_max):
            points.append((x1, y1))
        if (0 <= x2 <= x_max) and (0 <= y2 <= y_max):
            points.append((x2, y2))
        if (0 <= x3 <= x_max) and (0 <= y3 <= y_max):
            points.append((x3, y3))
        if (0 <= x4 <= x_max) and (0 <= y4 <= y_max):
            points.append((x4, y4))

        return points

    def _compute_landmarks(self, bounds: list[Sequence[Point]], inner: bool) -> list[Point]:
        landmarks = []
        for bound in bounds:
            landmarks += self._compute_complementary_landmarks(bound, inner)
        return landmarks

    def _compute_complementary_landmarks(self, bound: Sequence[Point], inner: bool) -> list[Point]:
        p = Profile(self.dm, bound[0], bound[1])
        p.compute_extrema()
        key_vals = [p.b1, p.b2] if inner else [p.t1, p.t2]
        lm = []
        for i in key_vals:
            if i == -1:
                logger.error("Cannot compute a landmark")
                continue
            x, y, _ = p._index2xyz(i)
            lm.append((x, y))
        return lm

    def _validate_landmark_points(self) -> bool:
        if len(self.landmarks) < self.points:
            logger.warning("Only %d/%d landmarks were computed", len(self.landmarks), self.points)
            if len(self.landmarks) < MIN_REQ_POINTS:
                logger.error("Not sufficient landmarks for fitting the elliptical model")
                return False
            else:
                logger.warning("Trying to fit the elliptical model anyways")
        return True

    def _compute_ellipse(self) -> None:
        x, y = map(np.array, zip(*self.landmarks, strict=True))
        xrange, yrange = (0, self.dm.x_count), (0, self.dm.y_count)
        self.a, self.b, self.cx, self.cy, self.theta = fit_elipse(x, y, xrange, yrange)

    def ellipse_patch(
        self, scale: float = 1.0, visual_config: EllipseVisualConfig | None = None
    ) -> Ellipse:
        visual_config = self.visual if visual_config is None else visual_config
        cx, cy = self.cx, self.cy
        da, db = 2 * self.a, 2 * self.b
        theta_degree = self.theta * 180 / np.pi
        if scale != 1.0:
            cx, cy, da, db = (scale * i for i in [cx, cy, da, db])
        return Ellipse(
            (cx, cy),
            da,
            db,
            angle=theta_degree,
            fill=visual_config.fill,
            color=visual_config.color,
            alpha=visual_config.alpha,
        )

    def max_profile(self) -> Profile:
        bound = self._compute_profile_bounds(self.theta, self.cx, self.cy)
        return Profile(self.dm, bound[0], bound[1])

    def params(self) -> tuple[float, float, float, float, float]:
        return self.a, self.b, self.cx, self.cy, self.theta

    def interior(self) -> np.ndarray:
        interior: np.ndarray = self._equation() <= 1
        return interior

    def rim(self) -> np.ndarray:
        rim: np.ndarray = np.isclose(self._equation(), 1, rtol=0, atol=RIM_TOLERANCE)
        return rim

    def _equation(self) -> np.ndarray:
        y, x = np.mgrid[: self.dm.y_count, : self.dm.x_count]
        cos, sin = np.cos(self.theta), np.sin(self.theta)
        x_hat = cos * (x - self.cx) + sin * (y - self.cy)
        y_hat = -sin * (x - self.cx) + cos * (y - self.cy)
        equation: np.ndarray = (x_hat / self.a) ** 2 + (y_hat / self.b) ** 2
        return equation
