from collections.abc import Callable

import pytest

from craterslab import craters
from craterslab.classification import SurfaceType
from craterslab.craters import Observable, Surface
from craterslab.sensors import DepthMap

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
