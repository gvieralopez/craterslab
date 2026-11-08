import logging

import numpy as np
import pytest
from matplotlib import pyplot as plt

from craterslab.ellipse import EllipseVisualConfig, EllipticalModel
from craterslab.profiles import Profile
from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile, slope_bounds


@pytest.fixture
def model(crater: DepthMap) -> EllipticalModel:
    return EllipticalModel(crater, 20)


def test_plot_2d(crater: DepthMap, model: EllipticalModel) -> None:
    plot_2D(crater, profile=model.max_profile(), ellipse=model, title="Top view")
    assert plt.gca().get_title() == "Top view"
    assert plt.gca().patches


def test_plot_2d_alone(crater: DepthMap) -> None:
    plot_2D(crater)
    assert not plt.gca().patches


def test_plot_3d(crater: DepthMap, model: EllipticalModel) -> None:
    plot_3D(
        crater,
        profile=model.max_profile(),
        ellipse=model,
        ellipse_config=EllipseVisualConfig(color="blue"),
    )
    assert plt.gcf().axes


def test_plot_3d_default_ellipse_config(crater: DepthMap, model: EllipticalModel) -> None:
    plot_3D(crater, ellipse=model)
    assert plt.gcf().axes


def test_plot_profile(crater: DepthMap) -> None:
    plot_profile(Profile(crater, (0, 50), (100, 50)))
    assert plt.gca().lines


def test_plot_profile_with_slopes(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    plot_profile(profile, draw_slopes=True)
    assert len(plt.gca().lines) == 3


def test_plot_profile_reports_invalid_slopes(
    resolution: SensorResolution, caplog: pytest.LogCaptureFixture
) -> None:
    profile = Profile(DepthMap(np.zeros((30, 30)), resolution), (0, 15), (29, 15))
    with caplog.at_level(logging.ERROR):
        plot_profile(profile, draw_slopes=True)
    assert "slopes failed" in caplog.text


def test_slope_bounds(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    _, z = slope_bounds(profile, np.poly1d([2.0, 1.0]), 0, 10)
    assert z[0] == pytest.approx(1.0)


def test_slope_bounds_without_a_model(crater: DepthMap, caplog: pytest.LogCaptureFixture) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    x, z = slope_bounds(profile, np.poly1d([0.0]), 0, 10)
    assert x == z
    assert "Slopes are invalid" in caplog.text
