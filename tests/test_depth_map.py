from pathlib import Path

import numpy as np
import pytest

from craterslab.depth_map import DepthMap, SensorResolution


def test_resolution_rejects_unknown_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        SensorResolution(1.0, 1.0, 1.0, "inch")


def test_resolution_xyz() -> None:
    assert SensorResolution(1.0, 2.0, 3.0, "cm").xyz() == (1.0, 2.0, 3.0)


def test_counts_follow_numpy_axes(resolution: SensorResolution) -> None:
    dm = DepthMap(np.zeros((3, 5)), resolution)
    assert (dm.x_count, dm.y_count) == (5, 3)


def test_subtract(resolution: SensorResolution) -> None:
    result = DepthMap(np.full((2, 2), 3.0), resolution) - DepthMap(np.ones((2, 2)), resolution)
    assert np.array_equal(result.map, np.full((2, 2), 2.0))


def test_subtract_rejects_other_types(resolution: SensorResolution) -> None:
    with pytest.raises(TypeError):
        _ = DepthMap(np.ones((2, 2)), resolution) - 1  # type: ignore[operator]


def test_subtract_rejects_other_shapes(resolution: SensorResolution) -> None:
    with pytest.raises(ValueError, match="same shape"):
        _ = DepthMap(np.ones((2, 2)), resolution) - DepthMap(np.ones((3, 3)), resolution)


def test_subtract_rejects_other_resolutions(resolution: SensorResolution) -> None:
    other = SensorResolution(2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="same resolution"):
        _ = DepthMap(np.ones((2, 2)), resolution) - DepthMap(np.ones((2, 2)), other)


def test_crop_with_padding(resolution: SensorResolution) -> None:
    dm = DepthMap(np.zeros((20, 20)), resolution)
    dm.crop((5, 5, 4, 3), padding=1)
    assert dm.map.shape == (5, 6)


def test_crop_borders(resolution: SensorResolution) -> None:
    dm = DepthMap(np.zeros((100, 100)), resolution)
    dm.crop_borders(0.5)
    assert dm.map.shape == (50, 50)


@pytest.mark.parametrize("ratio", [0, 1, -0.5, 1.5])
def test_crop_borders_rejects_bad_ratios(resolution: SensorResolution, ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio"):
        DepthMap(np.zeros((10, 10)), resolution).crop_borders(ratio)


def test_auto_crop_keeps_the_crater(crater: DepthMap) -> None:
    before = crater.map.shape
    crater.auto_crop()
    assert crater.map.shape[0] <= before[0]
    assert crater.map.min() == pytest.approx(-8.0, abs=0.1)


def test_heights_use_the_z_resolution() -> None:
    dm = DepthMap(np.array([[2.0, -4.0]]), SensorResolution(1.0, 1.0, 0.5))
    assert dm.heights.tolist() == [[1.0, -2.0]]


def test_auto_crop_keeps_a_surface_with_nothing_central(resolution: SensorResolution) -> None:
    surface = np.zeros((100, 100), dtype=np.float32)
    surface[2:14, 2:14] = 5
    dm = DepthMap(surface, resolution)
    dm.auto_crop()
    assert dm.map.shape == (100, 100)


def test_save_and_load(tmp_path: Path) -> None:
    original = DepthMap(np.arange(6.0).reshape(2, 3), SensorResolution(1.0, 2.0, 3.0, "cm"))
    path = tmp_path / "map.npz"
    original.save(path)
    loaded = DepthMap.load(path)
    assert np.array_equal(loaded.map, original.map)
    assert loaded.sensor.xyz() == (1.0, 2.0, 3.0)
    assert loaded.sensor.scale == "cm"
