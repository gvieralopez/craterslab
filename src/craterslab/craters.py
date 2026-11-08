import logging
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from craterslab.classification import Classifier, SurfaceType
from craterslab.ellipse import EllipseFit, EllipticalModel
from craterslab.profiles import Profile
from craterslab.sensors import DepthMap

logger = logging.getLogger(__name__)

# Groups of surfaces
CRATER_SURFACES = [SurfaceType.SIMPLE_CRATER, SurfaceType.COMPLEX_CRATER]
ALL_KNOWN_SURFACES = [*CRATER_SURFACES, SurfaceType.SAND_MOUND]

DEFAULT_ELLIPSE_POINTS = 20


@dataclass
class Observable:
    name: str
    symbol: str
    value: float
    units: str

    def __str__(self) -> str:
        return f"{self.name} ({self.symbol}): {self.value:.2f} {self.units}"


class Surface:
    """
    Describes a surface found in a Depth Map
    """

    def __init__(
        self,
        depth_map: DepthMap,
        surface_type: SurfaceType,
        ellipse_points: int,
        em: EllipticalModel | None,
        inner_ellipse: EllipticalModel | None,
    ) -> None:
        self.dm = depth_map
        self.ellipse_points = ellipse_points
        self.max_profile: Profile | None = None
        self.em = em
        self.inner_ellipse = inner_ellipse

        self.available_observables: dict[str, dict[str, Any]] = {
            "d_max": {"func": self.d_max, "compute_for": ALL_KNOWN_SURFACES},
            "d_exc": {"func": self.d_exc, "compute_for": ALL_KNOWN_SURFACES},
            "epsilon": {"func": self.epsilon, "compute_for": CRATER_SURFACES},
            "D": {"func": self.D, "compute_for": CRATER_SURFACES},
            "H_cp": {"func": self.H_cp, "compute_for": [SurfaceType.COMPLEX_CRATER]},
            "H_m": {"func": self.H_m, "compute_for": ALL_KNOWN_SURFACES},
            "mean_h_rim": {"func": self.mean_h_rim, "compute_for": CRATER_SURFACES},
            "V_in": {"func": self.V_in, "compute_for": CRATER_SURFACES},
            "V_ex": {"func": self.V_ex, "compute_for": CRATER_SURFACES},
            "V_exc": {"func": self.V_exc, "compute_for": ALL_KNOWN_SURFACES},
            "V_cp": {"func": self.V_cp, "compute_for": [SurfaceType.COMPLEX_CRATER]},
        }

        self.set_type(surface_type)

    @classmethod
    def from_depth_map(
        cls,
        depth_map: DepthMap,
        surface_type: SurfaceType = SurfaceType.UNKNOWN,
        ellipse_points: int = DEFAULT_ELLIPSE_POINTS,
    ) -> "Surface":
        return cls(depth_map, surface_type, ellipse_points, None, None)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Surface":
        """Load a saved surface. A plain depth map file gives an Unknown surface."""
        depth_map = DepthMap.load(path)
        with np.load(path) as data:
            surface_type = SurfaceType.UNKNOWN
            ellipse_points = DEFAULT_ELLIPSE_POINTS
            if "surface_type" in data:
                surface_type = SurfaceType[str(data["surface_type"])]
                ellipse_points = int(data["ellipse_points"])
            em = cls._load_ellipse(data, "ellipse", depth_map, ellipse_points)
            inner_ellipse = cls._load_ellipse(data, "inner_ellipse", depth_map, ellipse_points)
        return cls(depth_map, surface_type, ellipse_points, em, inner_ellipse)

    @staticmethod
    def _load_ellipse(
        data: Any, key: str, depth_map: DepthMap, ellipse_points: int
    ) -> EllipticalModel | None:
        if key not in data:
            return None
        a, b, cx, cy, theta = data[key].tolist()
        landmarks = [(x, y) for x, y in data[f"{key}_landmarks"].tolist()]
        fit = EllipseFit(a, b, cx, cy, theta, landmarks)
        return EllipticalModel(depth_map, ellipse_points, fit=fit)

    def save(self, path: pathlib.Path | str) -> None:
        arrays = self.dm.to_arrays()
        arrays["surface_type"] = self.type.name
        arrays["ellipse_points"] = self.ellipse_points
        for key, model in (("ellipse", self.em), ("inner_ellipse", self.inner_ellipse)):
            if model is not None:
                arrays[key] = np.array(model.params())
                arrays[f"{key}_landmarks"] = np.array(model.landmarks)
        np.savez(path, **arrays)

    @staticmethod
    def _fitted(model: EllipticalModel | None) -> EllipticalModel:
        if model is None:
            raise ValueError("This value needs a surface with a known type")
        return model

    def __repr__(self) -> str:
        output = "\n".join([str(o) for o_id, o in self.observables.items()])
        return f"\nFound: {self.type}\n\n{output}"

    def set_type(self, surface_type: SurfaceType) -> None:
        self.type = surface_type
        self._fit_ellipses()
        self.observables = self.compute_observables()

    def classify(self, classifier: Classifier) -> None:
        self.set_type(classifier.classify(self.dm))

    def _fit_ellipses(self) -> None:
        if self.type is SurfaceType.UNKNOWN:
            self.em = self.inner_ellipse = self.max_profile = None
            return
        if self.em is None:
            self.em = EllipticalModel(self.dm, self.ellipse_points)
        self.max_profile = self.em.max_profile()
        if self.type is SurfaceType.COMPLEX_CRATER and self.inner_ellipse is None:
            self.inner_ellipse = EllipticalModel(self.dm, self.ellipse_points, inner=True)

    def compute_observables(self) -> dict[str, Observable]:
        observables = {}
        for o_id, details in self.available_observables.items():
            if self.type in details["compute_for"]:
                observables[o_id] = details["func"]()
        return observables

    def _d_exc(self) -> float:
        return float(np.min(self.dm.heights))

    def _H_max(self) -> float:
        return float(np.max(self.dm.heights))

    def _ellipse(self) -> np.ndarray:
        # create a meshgrid of x and y coordinates
        x, y = np.ogrid[: self.dm.x_count, : self.dm.y_count]

        # calculate the ellipse equation for each point in the grid
        a, b, x0, y0, theta = self._fitted(self.em).params()
        x_rot = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)  # rotate x coordinates
        y_rot = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)  # rotate y coordinates
        result: np.ndarray = ((x_rot) ** 2 / a**2) + ((y_rot) ** 2 / b**2)
        return result

    def _ellipse_perimeter(self) -> np.ndarray:
        perimeter: np.ndarray = np.isclose(self._ellipse(), 1, rtol=0, atol=0.05).T
        return perimeter

    def _ellipse_content(self) -> np.ndarray:
        return np.transpose(self._ellipse() <= 1)

    def _inner_z_values(self) -> np.ndarray:
        a, b, cx, cy, _ = self._fitted(self.inner_ellipse).params()
        rows, cols = self.dm.map.shape
        x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

        # Translate the grid coordinates to the ellipse center
        x_indices -= int(cx)
        y_indices -= int(cy)

        # Compute the ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        ellipse_mask = (x_indices**2 / a**2) + (y_indices**2 / b**2) <= 1

        # Extract the points inside the ellipse from the ndarray
        inner_values: np.ndarray = self.dm.heights[ellipse_mask]
        return inner_values

    def _h_rim(self) -> np.ndarray:
        heights: np.ndarray = self.dm.heights[self._ellipse_perimeter()]
        return heights

    def _mean_h_rim(self) -> float:
        return float(np.mean(self._h_rim()))

    def _V_in(self) -> float:
        h_max, h_min = self._mean_h_rim(), self._d_exc()
        inner_values = self.dm.heights[self._ellipse_content()] - h_min
        positive_sum = np.sum(inner_values[inner_values <= h_max - h_min])
        return float(positive_sum * self.dm.x_res * self.dm.y_res)

    def _H_cp(self) -> float:
        points_inside_ellipse = self._inner_z_values()
        max_val = np.max(points_inside_ellipse)
        return float(max_val - self._d_exc())

    def _V_cp(self) -> float:
        points_inside_ellipse = self._inner_z_values()
        points_inside_ellipse -= np.min(points_inside_ellipse)
        total_sum = np.sum(points_inside_ellipse)
        return float(total_sum * self.dm.x_res * self.dm.y_res)

    def _V_ex(self) -> float:
        inner_values = self.dm.heights[self._ellipse_content()]
        negative_sum = -np.sum(inner_values[inner_values < 0])
        return float(negative_sum * self.dm.x_res * self.dm.y_res)

    def d_max(self) -> Observable:
        val = abs(self._d_exc()) + abs(self._mean_h_rim())
        units = self.dm.sensor.scale
        return Observable("Apparent Depth", "d_max", val, units)

    def d_exc(self) -> Observable:
        val = self._d_exc()
        units = self.dm.sensor.scale
        return Observable("Max Excavated Depth", "d_exc", val, units)

    def epsilon(self) -> Observable:
        em = self._fitted(self.em)
        a, b = em.a, em.b
        val = math.sqrt(a**2 - b**2) / a
        units = ""
        return Observable("Eccentricity", "epsilon", val, units)

    def D(self) -> Observable:
        val = self._fitted(self.em).a * 2 * self.dm.x_res
        units = self.dm.sensor.scale
        return Observable("Diameter", "D", val, units)

    def V_cp(self) -> Observable:
        val = self._V_cp()
        units = f"{self.dm.sensor.scale}³"
        return Observable("Volume of central peak", "V_cp", val, units)

    def H_cp(self) -> Observable:
        val = self._H_cp()
        units = self.dm.sensor.scale
        return Observable("Heigh of central peak", "H_cp", val, units)

    def H_m(self) -> Observable:
        val = self._H_max()
        units = self.dm.sensor.scale
        return Observable("Maximum heigh", "H_m", val, units)

    def mean_h_rim(self) -> Observable:
        val = self._mean_h_rim()
        units = self.dm.sensor.scale
        return Observable("Mean Heigh over the rim", "mean_h_rim", val, units)

    def V_in(self) -> Observable:
        val = self._V_in()
        units = f"{self.dm.sensor.scale}³"
        return Observable("Concavity Volume", "V_in", val, units)

    def V_ex(self) -> Observable:
        val = self._V_ex()
        units = f"{self.dm.sensor.scale}³"
        return Observable("Excavated Volume", "V_ex", val, units)

    def V_exc(self) -> Observable:
        positive_sum = np.sum(self.dm.heights[self.dm.heights > 0])
        val = positive_sum * self.dm.x_res * self.dm.y_res
        units = f"{self.dm.sensor.scale}³"
        return Observable("Excess Volume", "V_exc", val, units)
