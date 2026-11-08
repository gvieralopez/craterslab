import logging

import numpy as np
import pytest

from craterslab.ellipse import EllipseFit, EllipseVisualConfig, EllipticalModel, fit_elipse
from craterslab.sensors import DepthMap, SensorResolution


def _ellipse_points(a: float, b: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return cx + a * np.cos(t), cy + b * np.sin(t)


def test_fit_recovers_an_ellipse() -> None:
    a, b, cx, cy, _ = fit_elipse(*_ellipse_points(30, 20, 50, 50), (0, 100), (0, 100))
    assert (a, b, cx, cy) == pytest.approx((30, 20, 50, 50), abs=0.5)


def test_fit_orders_the_axes() -> None:
    a, b, _, _, _ = fit_elipse(*_ellipse_points(20, 30, 50, 50), (0, 100), (0, 100))
    assert a > b


@pytest.mark.parametrize(("points", "message"), [(5, "even"), (2, "greater")])
def test_rejects_bad_point_counts(crater: DepthMap, points: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EllipticalModel(crater, points)


def test_fits_the_rim_of_a_crater(crater: DepthMap) -> None:
    model = EllipticalModel(crater, 20)
    a, b, cx, cy, _ = model.params()
    assert (a, b, cx, cy) == pytest.approx((30, 30, 50, 50), abs=3)
    assert len(model.landmarks) == 20


def test_inner_landmarks_are_inside_the_rim(crater: DepthMap) -> None:
    inner = EllipticalModel(crater, 20, inner=True)
    assert all(np.hypot(x - 50, y - 50) < 30 for x, y in inner.landmarks)


def test_too_few_landmarks_raise(
    resolution: SensorResolution, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="landmarks"):
        EllipticalModel(DepthMap(np.zeros((30, 30)), resolution), 20)
    assert "Not sufficient landmarks" in caplog.text


def test_partial_landmarks_still_fit(crater: DepthMap, caplog: pytest.LogCaptureFixture) -> None:
    model = EllipticalModel(crater, 20)
    model.landmarks = model.landmarks[:10]
    with caplog.at_level(logging.WARNING):
        assert model._validate_landmark_points()
    assert "10/20 landmarks were computed" in caplog.text


def test_patch_scales_and_uses_the_visual_config(crater: DepthMap) -> None:
    model = EllipticalModel(crater, 20)
    patch = model.ellipse_patch(scale=2.0, visual_config=EllipseVisualConfig(color="blue"))
    assert patch.width == pytest.approx(4 * model.a)
    assert patch.center == pytest.approx((2 * model.cx, 2 * model.cy))


def test_default_patch(crater: DepthMap) -> None:
    model = EllipticalModel(crater, 20)
    assert model.ellipse_patch().width == pytest.approx(2 * model.a)


def test_max_profile(crater: DepthMap) -> None:
    profile = EllipticalModel(crater, 20).max_profile()
    assert len(profile) == 1000


def test_a_given_fit_is_used_as_it_is(crater: DepthMap) -> None:
    fit = EllipseFit(5.0, 4.0, 50.0, 51.0, 0.1, [(1.0, 2.0)])
    model = EllipticalModel(crater, 20, fit=fit)
    assert model.params() == (5.0, 4.0, 50.0, 51.0, 0.1)
    assert model.landmarks == [(1.0, 2.0)]
