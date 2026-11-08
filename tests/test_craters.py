from collections.abc import Callable

import pytest

from craterslab import craters
from craterslab.classification import SurfaceType
from craterslab.craters import Observable, Surface
from craterslab.sensors import DepthMap, SensorResolution

COMMON = {"d_max", "d_exc", "H_m", "V_exc"}
CRATER = COMMON | {"epsilon", "D", "mean_h_rim", "V_in", "V_ex"}


def test_observable_text() -> None:
    assert str(Observable("Diameter", "D", 1.234, "mm")) == "Diameter (D): 1.23 mm"


@pytest.mark.parametrize(
    ("name", "surface_type", "observables"),
    [
        ("fluidized_1", SurfaceType.SIMPLE_CRATER, CRATER),
        ("fluidized_10", SurfaceType.COMPLEX_CRATER, CRATER | {"H_cp", "V_cp"}),
        ("compacted_1", SurfaceType.SAND_MOUND, COMMON),
        ("plane_10", SurfaceType.UNKNOWN, set()),
    ],
)
def test_real_surfaces(
    real_map: Callable[[str], DepthMap], name: str, surface_type: SurfaceType, observables: set[str]
) -> None:
    surface = Surface(real_map(name))
    assert surface.type is surface_type
    assert set(surface.observables) == observables
    assert str(surface_type) in repr(surface)


def test_real_simple_crater_values(real_map: Callable[[str], DepthMap]) -> None:
    dm = real_map("fluidized_1")
    surface = Surface(dm)
    rim = surface.observables["mean_h_rim"].value
    assert surface.observables["d_max"].value == pytest.approx(abs(dm.map.min()) + abs(rim))
    assert surface.observables["D"].value == pytest.approx(2 * surface.em.a * dm.x_res)
    assert 0 <= surface.observables["epsilon"].value < 1


def test_set_type_recomputes_observables(real_map: Callable[[str], DepthMap]) -> None:
    surface = Surface(real_map("fluidized_10"))
    surface.set_type(SurfaceType.SAND_MOUND)
    assert set(surface.observables) == COMMON


def test_missing_model_falls_back_to_simple_crater(
    crater: DepthMap, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(craters, "get_trained_model", lambda: None)
    assert Surface(crater).type is SurfaceType.SIMPLE_CRATER
    assert "Using default class" in caplog.text


def _rescaled(dm: DepthMap, factor: float) -> DepthMap:
    """The same physical surface, sampled with a z resolution `factor` times finer."""
    sensor = SensorResolution(dm.x_res, dm.y_res, dm.z_res / factor, dm.sensor.scale)
    return DepthMap(dm.map * factor, sensor)


@pytest.mark.parametrize("name", ["_V_in", "_V_ex", "_V_cp", "_H_cp", "_mean_h_rim", "_d_exc"])
def test_values_do_not_depend_on_the_z_resolution(crater: DepthMap, name: str) -> None:
    surface = Surface(crater)
    surface.set_type(SurfaceType.COMPLEX_CRATER)
    before = getattr(surface, name)()
    surface.dm = _rescaled(crater, 2.0)
    surface.inner_ellipse.dm = surface.dm
    assert getattr(surface, name)() == pytest.approx(before)


def test_excess_volume_uses_all_three_resolutions(crater: DepthMap) -> None:
    surface = Surface(crater)
    block = crater.map * 0
    block[:4, :5] = 8.0
    surface.dm = DepthMap(block, SensorResolution(2.0, 3.0, 0.5, "cm"))
    assert surface.V_exc().value == pytest.approx(20 * 4.0 * 6.0)


def test_maximum_height_has_its_own_symbol(crater: DepthMap) -> None:
    assert Surface(crater).H_m().symbol == "H_m"
