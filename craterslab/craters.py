import math
import logging
from dataclasses import dataclass

import numpy as np

from craterslab.classification import SurfaceType, get_trained_model, normalize
from craterslab.ellipse import EllipticalModel
from craterslab.sensors import DepthMap

# Groups of surfaces
CRATER_SURFACES = [SurfaceType.SIMPLE_CRATER, SurfaceType.COMPLEX_CRATER]
ALL_KNOWN_SURFACES = CRATER_SURFACES + [SurfaceType.SAND_MOUND]


@dataclass
class Observable:
    name: str
    symbol: str
    value: float
    units: str

    def __str__(self):
        return f"{self.name} ({self.symbol}): {self.value:.2f} {self.units}"


class Surface:
    """
    Describes a surface found in a Depth Map
    """

    def __init__(self, depth_map: DepthMap, ellipse_points: int = 20) -> None:
        self.dm = depth_map
        self.em = EllipticalModel(depth_map, ellipse_points)
        self.max_profile = self.em.max_profile()
        self.type = self.classify()

        self.available_observables = {
            "d_max": {"func": self.d_max, "compute_for": ALL_KNOWN_SURFACES},
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

        self.init_observables()

    def init_observables(self):
        if self.type == SurfaceType.COMPLEX_CRATER:
            self.inner_ellipse = EllipticalModel(self.dm, self.em.points, inner=True)
        self.observables = self.compute_observables()

    def __repr__(self) -> str:
        output = "\n".join([str(o) for o_id, o in self.observables.items()])
        return f"\nFound: {self.type}\n\n{output}"

    def set_type(self, surface_type: SurfaceType):
        self.type = surface_type
        self.init_observables()

    def classify(self) -> SurfaceType:
        classifier = get_trained_model()
        if classifier is not None:
            img = normalize(self.dm.map)
            class_id = np.argmax(classifier.predict(img))
            return SurfaceType(class_id)
        logging.warning(f"Using default class {SurfaceType(1)}")
        return SurfaceType(1)

    def compute_observables(self) -> dict[str, Observable]:
        observables = {}
        for o_id, details in self.available_observables.items():
            if self.type in details["compute_for"]:
                observables[o_id] = details["func"]()
        return observables

    def _d_max(self) -> float:
        return np.min(self.dm.map) * self.dm.z_res

    def _H_max(self) -> float:
        return np.max(self.dm.map) * self.dm.z_res

    def _ellipse(self) -> np.ndarray:
        # create a meshgrid of x and y coordinates
        x, y = np.ogrid[: self.dm.x_count, : self.dm.y_count]

        # calculate the ellipse equation for each point in the grid
        a, b, x0, y0, theta = self.em.params()
        x_rot = (x - x0) * np.cos(theta) - (y - y0) * np.sin(
            theta
        )  # rotate x coordinates
        y_rot = (x - x0) * np.sin(theta) + (y - y0) * np.cos(
            theta
        )  # rotate y coordinates
        return ((x_rot) ** 2 / a**2) + ((y_rot) ** 2 / b**2)

    def _ellipse_perimeter(self) -> np.ndarray[bool]:
        return np.isclose(self._ellipse(), 1, rtol=0, atol=0.05).T

    def _ellipse_content(self) -> np.ndarray[bool]:
        return np.transpose(self._ellipse() <= 1)

    def _inner_z_values(self) -> np.array:
        a, b, cx, cy, _ = self.inner_ellipse.params()
        M, N = self.dm.map.shape
        x_indices, y_indices = np.meshgrid(np.arange(N), np.arange(M))

        # Translate the grid coordinates to the ellipse center
        x_indices -= int(cx)
        y_indices -= int(cy)

        # Compute the ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        ellipse_mask = (x_indices**2 / a**2) + (y_indices**2 / b**2) <= 1

        # Extract the points inside the ellipse from the ndarray
        return self.dm.map[ellipse_mask]

    def _h_rim(self):
        return self.dm.map[self._ellipse_perimeter()] * self.dm.z_res

    def _mean_h_rim(self) -> float:
        return np.mean(self._h_rim())

    def _V_in(self) -> float:
        h_max, h_min = self._mean_h_rim(), self._d_max()
        inner_values = self.dm.map[self._ellipse_content()] - h_min
        positive_sum = np.sum(inner_values[inner_values <= h_max - h_min])
        return positive_sum * self.dm.x_res * self.dm.y_res 

    def _H_cp(self) -> float:
        points_inside_ellipse = self._inner_z_values()
        max_val = np.max(points_inside_ellipse) * self.dm.z_res
        return max_val - self._d_max()
    
    def _V_cp(self) -> float:
        points_inside_ellipse = self._inner_z_values() * self.dm.z_res
        points_inside_ellipse -=  np.min(points_inside_ellipse)
        total_sum = np.sum(points_inside_ellipse)
        return total_sum * self.dm.x_res * self.dm.y_res   

    def _V_ex(self) -> float:
        inner_values = self.dm.map[self._ellipse_content()]
        negative_sum = -np.sum(inner_values[inner_values < 0])
        return negative_sum * self.dm.x_res * self.dm.y_res

    def d_max(self) -> Observable:
        val = self._d_max()
        units = self.dm.sensor.scale
        return Observable("Apparent Depth", "d_max", val, units)

    def epsilon(self) -> Observable:
        a, b = self.em.a, self.em.b
        val = math.sqrt(a**2 - b**2) / a
        units = ""
        return Observable("Eccentricity", "epsilon", val, units)

    def D(self) -> Observable:
        val = self.em.a * 2 * self.dm.x_res
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
        return Observable("Maximum heigh", "H_cp", val, units)

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
        positive_sum = np.sum(self.dm.map[self.dm.map > 0])
        val = positive_sum * self.dm.x_res * self.dm.y_res
        units = f"{self.dm.sensor.scale}³"
        return Observable("Excess Volume", "V_exc", val, units)
