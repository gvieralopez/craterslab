from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from craterslab import craters
from craterslab.classification import SurfaceType
from craterslab.craters import Observable, Surface
from craterslab.ellipse import EllipseFit, EllipticalModel
from craterslab.sensors import DepthMap, SensorResolution

COMMON = {"d_max", "d_exc", "H_m", "V_exc"}
CRATER = COMMON | {"epsilon", "D", "mean_h_rim", "V_in", "V_ex"}
COMPLEX = CRATER | {"H_cp", "V_cp"}


class FixedClassifier:
    def __init__(self, answer: SurfaceType) -> None:
        self.answer = answer
        self.seen: list[DepthMap] = []

    def classify(self, depth_map: DepthMap) -> SurfaceType:
        self.seen.append(depth_map)
        return self.answer


def test_observable_text() -> None:
    assert str(Observable("Diameter", "D", 1.234, "mm")) == "Diameter (D): 1.23 mm"


def test_a_new_surface_is_unknown(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater)
    assert surface.type is SurfaceType.UNKNOWN
    assert surface.em is None
    assert surface.max_profile is None
    assert surface.observables == {}
    assert "Unknown" in repr(surface)


def test_an_unknown_flat_surface_is_fine(resolution: SensorResolution) -> None:
    surface = Surface.from_depth_map(DepthMap(np.zeros((60, 60)), resolution))
    assert surface.observables == {}


def test_a_flat_surface_cannot_get_a_known_type(resolution: SensorResolution) -> None:
    flat = DepthMap(np.zeros((60, 60)), resolution)
    with pytest.raises(ValueError, match="landmarks"):
        Surface.from_depth_map(flat, SurfaceType.SIMPLE_CRATER)


def test_values_of_an_unknown_surface_need_a_type(crater: DepthMap) -> None:
    with pytest.raises(ValueError, match="known type"):
        craters._diameter(Surface.from_depth_map(crater))


@pytest.mark.parametrize(
    ("name", "surface_type", "observables"),
    [
        ("fluidized_1", SurfaceType.SIMPLE_CRATER, CRATER),
        ("fluidized_10", SurfaceType.COMPLEX_CRATER, COMPLEX),
        ("compacted_1", SurfaceType.SAND_MOUND, COMMON),
        ("plane_10", SurfaceType.UNKNOWN, set()),
    ],
)
def test_observables_follow_the_type(
    real_map: Callable[[str], DepthMap], name: str, surface_type: SurfaceType, observables: set[str]
) -> None:
    surface = Surface.from_depth_map(real_map(name), surface_type)
    assert surface.type is surface_type
    assert set(surface.observables) == observables
    assert str(surface_type) in repr(surface)


def test_real_simple_crater_values(real_map: Callable[[str], DepthMap]) -> None:
    dm = real_map("fluidized_1")
    surface = Surface.from_depth_map(dm, SurfaceType.SIMPLE_CRATER)
    rim = surface.observables["mean_h_rim"].value
    assert surface.em is not None
    assert surface.observables["d_max"].value == pytest.approx(abs(dm.map.min()) + abs(rim))
    assert surface.observables["D"].value == pytest.approx(2 * surface.em.a * dm.x_res)
    assert 0 <= surface.observables["epsilon"].value < 1


def test_classify_uses_the_classifier_answer(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater)
    classifier = FixedClassifier(SurfaceType.COMPLEX_CRATER)
    surface.classify(classifier)
    assert classifier.seen == [crater]
    assert surface.type is SurfaceType.COMPLEX_CRATER
    assert set(surface.observables) == COMPLEX


def test_set_type_recomputes_observables(real_map: Callable[[str], DepthMap]) -> None:
    surface = Surface.from_depth_map(real_map("fluidized_10"), SurfaceType.COMPLEX_CRATER)
    surface.set_type(SurfaceType.SAND_MOUND)
    assert set(surface.observables) == COMMON


def test_going_back_to_unknown_clears_the_models(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.COMPLEX_CRATER)
    surface.set_type(SurfaceType.UNKNOWN)
    assert surface.em is None
    assert surface.inner_ellipse is None
    assert surface.max_profile is None
    assert surface.observables == {}


def test_save_and_load_unknown(crater: DepthMap, tmp_path: Path) -> None:
    path = tmp_path / "surface.npz"
    Surface.from_depth_map(crater).save(path)
    loaded = Surface.load(path)
    assert loaded.type is SurfaceType.UNKNOWN
    assert loaded.em is None
    assert np.array_equal(loaded.dm.map, crater.map)


@pytest.mark.parametrize("surface_type", [SurfaceType.SIMPLE_CRATER, SurfaceType.COMPLEX_CRATER])
def test_save_and_load_gives_the_same_surface(
    real_map: Callable[[str], DepthMap], tmp_path: Path, surface_type: SurfaceType
) -> None:
    path = tmp_path / "surface.npz"
    original = Surface.from_depth_map(real_map("fluidized_10"), surface_type, ellipse_points=12)
    original.save(path)
    loaded = Surface.load(path)
    assert loaded.type is surface_type
    assert loaded.ellipse_points == 12
    assert loaded.em is not None
    assert original.em is not None
    assert loaded.em.params() == original.em.params()
    assert loaded.em.landmarks == original.em.landmarks
    assert {k: o.value for k, o in loaded.observables.items()} == {
        k: o.value for k, o in original.observables.items()
    }
    assert str(loaded) == str(original)


def test_load_does_not_fit_again(
    crater: DepthMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.npz"
    Surface.from_depth_map(crater, SurfaceType.COMPLEX_CRATER).save(path)

    def fail(*args: object) -> None:
        raise AssertionError("the ellipse was fitted again")

    monkeypatch.setattr(EllipticalModel, "_compute_ellipse", fail)
    assert Surface.load(path).type is SurfaceType.COMPLEX_CRATER


def test_a_saved_surface_can_be_read_as_a_depth_map(crater: DepthMap, tmp_path: Path) -> None:
    path = tmp_path / "surface.npz"
    Surface.from_depth_map(crater, SurfaceType.SIMPLE_CRATER).save(path)
    assert np.array_equal(DepthMap.load(path).map, crater.map)


def test_a_depth_map_file_loads_as_unknown(crater: DepthMap, tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    crater.save(path)
    assert Surface.load(path).type is SurfaceType.UNKNOWN


def test_a_loaded_surface_can_change_type(crater: DepthMap, tmp_path: Path) -> None:
    path = tmp_path / "surface.npz"
    Surface.from_depth_map(crater, SurfaceType.SIMPLE_CRATER).save(path)
    loaded = Surface.load(path)
    loaded.set_type(SurfaceType.COMPLEX_CRATER)
    assert set(loaded.observables) == COMPLEX


def _rescaled(dm: DepthMap, factor: float) -> DepthMap:
    """The same physical surface, sampled with a z resolution `factor` times finer."""
    sensor = SensorResolution(dm.x_res, dm.y_res, dm.z_res / factor, dm.sensor.scale)
    return DepthMap(dm.map * factor, sensor)


@pytest.mark.parametrize("symbol", sorted(COMPLEX))
def test_values_do_not_depend_on_the_z_resolution(crater: DepthMap, symbol: str) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.COMPLEX_CRATER)
    before = surface.observables[symbol].value
    surface.dm = _rescaled(crater, 2.0)
    assert surface.compute_observables()[symbol].value == pytest.approx(before)


def test_excess_volume_uses_all_three_resolutions(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.SIMPLE_CRATER)
    block = crater.map * 0
    block[:4, :5] = 8.0
    surface.dm = DepthMap(block, SensorResolution(2.0, 3.0, 0.5, "cm"))
    assert surface.compute_observables()["V_exc"].value == pytest.approx(20 * 4.0 * 6.0)


def test_maximum_height_has_its_own_symbol(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.SIMPLE_CRATER)
    assert surface.observables["H_m"].symbol == "H_m"


def test_every_observable_is_defined_once() -> None:
    symbols = [definition.symbol for definition in craters.OBSERVABLES]
    assert len(symbols) == len(set(symbols))
    assert set(symbols) == COMPLEX


@pytest.mark.parametrize(("symbol", "units"), [("epsilon", ""), ("D", "mm"), ("V_in", "mm³")])
def test_units_follow_the_sensor_scale(crater: DepthMap, symbol: str, units: str) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.SIMPLE_CRATER)
    assert surface.observables[symbol].units == units


def test_observables_are_listed_in_table_order(crater: DepthMap) -> None:
    surface = Surface.from_depth_map(crater, SurfaceType.COMPLEX_CRATER)
    assert list(surface.observables) == [d.symbol for d in craters.OBSERVABLES]


def _tilted_fit() -> EllipseFit:
    return EllipseFit(40.0, 18.0, 60.0, 55.0, 0.6, [(0.0, 0.0)])


def test_volumes_follow_a_tilted_ellipse(resolution: SensorResolution) -> None:
    depth_map = DepthMap(np.zeros((120, 120)), resolution)
    em = EllipticalModel(depth_map, 20, fit=_tilted_fit())
    depth_map.map = np.where(em.interior(), -1.0, 0.0)
    surface = Surface(depth_map, SurfaceType.SIMPLE_CRATER, 20, em, None)
    assert surface.observables["V_ex"].value == pytest.approx(em.interior().sum())


def test_the_rim_height_follows_a_tilted_ellipse(resolution: SensorResolution) -> None:
    depth_map = DepthMap(np.zeros((120, 120)), resolution)
    em = EllipticalModel(depth_map, 20, fit=_tilted_fit())
    depth_map.map = np.where(em.rim(), 3.0, 0.0)
    surface = Surface(depth_map, SurfaceType.SIMPLE_CRATER, 20, em, None)
    assert surface.observables["mean_h_rim"].value == pytest.approx(3.0)


def test_the_central_peak_follows_a_tilted_ellipse(resolution: SensorResolution) -> None:
    depth_map = DepthMap(np.zeros((120, 120)), resolution)
    em = EllipticalModel(depth_map, 20, fit=_tilted_fit())
    heights = np.zeros((120, 120))
    heights[55 + round(35 * np.sin(0.6)), 60 + round(35 * np.cos(0.6))] = 7.0
    depth_map.map = heights
    surface = Surface(depth_map, SurfaceType.COMPLEX_CRATER, 20, em, em)
    assert surface.observables["H_cp"].value == pytest.approx(7.0)
