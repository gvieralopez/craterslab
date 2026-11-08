import math
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from craterslab.classification import Classifier, SurfaceType
from craterslab.ellipse import EllipseFit, EllipticalModel
from craterslab.profiles import Profile
from craterslab.sensors import DepthMap

# Groups of surfaces
COMPLEX_SURFACES = (SurfaceType.COMPLEX_CRATER,)
CRATER_SURFACES = (SurfaceType.SIMPLE_CRATER, *COMPLEX_SURFACES)
ALL_KNOWN_SURFACES = (*CRATER_SURFACES, SurfaceType.SAND_MOUND)

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
        return {
            definition.symbol: definition.measure(self)
            for definition in OBSERVABLES
            if self.type in definition.surfaces
        }


def _fitted(model: EllipticalModel | None) -> EllipticalModel:
    if model is None:
        raise ValueError("This value needs a surface with a known type")
    return model


def _excavated_depth(surface: Surface) -> float:
    return float(np.min(surface.dm.heights))


def _maximum_height(surface: Surface) -> float:
    return float(np.max(surface.dm.heights))


def _mean_rim_height(surface: Surface) -> float:
    return float(np.mean(surface.dm.heights[_fitted(surface.em).rim()]))


def _apparent_depth(surface: Surface) -> float:
    return abs(_excavated_depth(surface)) + abs(_mean_rim_height(surface))


def _eccentricity(surface: Surface) -> float:
    em = _fitted(surface.em)
    return math.sqrt(em.a**2 - em.b**2) / em.a


def _diameter(surface: Surface) -> float:
    return float(_fitted(surface.em).a * 2 * surface.dm.x_res)


def _central_peak_values(surface: Surface) -> np.ndarray:
    values: np.ndarray = surface.dm.heights[_fitted(surface.inner_ellipse).interior()]
    return values


def _central_peak_height(surface: Surface) -> float:
    return float(np.max(_central_peak_values(surface)) - _excavated_depth(surface))


def _central_peak_volume(surface: Surface) -> float:
    values = _central_peak_values(surface)
    return float(np.sum(values - np.min(values)) * surface.dm.x_res * surface.dm.y_res)


def _concavity_volume(surface: Surface) -> float:
    rim, bottom = _mean_rim_height(surface), _excavated_depth(surface)
    inner_values = surface.dm.heights[_fitted(surface.em).interior()] - bottom
    below_rim = np.sum(inner_values[inner_values <= rim - bottom])
    return float(below_rim * surface.dm.x_res * surface.dm.y_res)


def _excavated_volume(surface: Surface) -> float:
    inner_values = surface.dm.heights[_fitted(surface.em).interior()]
    return float(-np.sum(inner_values[inner_values < 0]) * surface.dm.x_res * surface.dm.y_res)


def _excess_volume(surface: Surface) -> float:
    heights = surface.dm.heights
    return float(np.sum(heights[heights > 0]) * surface.dm.x_res * surface.dm.y_res)


@dataclass(frozen=True, slots=True)
class ObservableDefinition:
    symbol: str
    name: str
    units_power: int
    surfaces: tuple[SurfaceType, ...]
    compute: Callable[[Surface], float]

    def measure(self, surface: Surface) -> Observable:
        return Observable(
            self.name, self.symbol, self.compute(surface), self._units(surface.dm.sensor.scale)
        )

    def _units(self, scale: str) -> str:
        return {0: "", 1: f"{scale}", 3: f"{scale}³"}[self.units_power]


OBSERVABLES = (
    ObservableDefinition("d_max", "Apparent Depth", 1, ALL_KNOWN_SURFACES, _apparent_depth),
    ObservableDefinition("d_exc", "Max Excavated Depth", 1, ALL_KNOWN_SURFACES, _excavated_depth),
    ObservableDefinition("epsilon", "Eccentricity", 0, CRATER_SURFACES, _eccentricity),
    ObservableDefinition("D", "Diameter", 1, CRATER_SURFACES, _diameter),
    ObservableDefinition(
        "H_cp", "Height of central peak", 1, COMPLEX_SURFACES, _central_peak_height
    ),
    ObservableDefinition("H_m", "Maximum height", 1, ALL_KNOWN_SURFACES, _maximum_height),
    ObservableDefinition(
        "mean_h_rim", "Mean Height over the rim", 1, CRATER_SURFACES, _mean_rim_height
    ),
    ObservableDefinition("V_in", "Concavity Volume", 3, CRATER_SURFACES, _concavity_volume),
    ObservableDefinition("V_ex", "Excavated Volume", 3, CRATER_SURFACES, _excavated_volume),
    ObservableDefinition("V_exc", "Excess Volume", 3, ALL_KNOWN_SURFACES, _excess_volume),
    ObservableDefinition(
        "V_cp", "Volume of central peak", 3, COMPLEX_SURFACES, _central_peak_volume
    ),
)
