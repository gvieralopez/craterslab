from pathlib import Path

import numpy as np
import pytest
import scipy.io

from craterslab.sensors import DepthMap, SensorResolution


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


def test_from_image_file_is_not_available(resolution: SensorResolution) -> None:
    with pytest.raises(NotImplementedError):
        DepthMap.from_image_file(np.zeros((2, 2)))


def test_kinect_without_driver(resolution: SensorResolution) -> None:
    assert DepthMap.kinect_driver_available() is False
    with pytest.raises(NotImplementedError):
        DepthMap.from_kinect_sensor(resolution)


def _write_mat(folder: Path) -> np.ndarray:
    samples = np.arange(16, dtype=float).reshape((2, 2, 1, 4))
    scipy.io.savemat(folder / "scan.mat", {"scan": samples, "other": samples * 2})
    return samples


def test_from_mat_file_averages_the_samples(tmp_path: Path) -> None:
    samples = _write_mat(tmp_path)
    dm = DepthMap.from_mat_file("scan.mat", data_folder=str(tmp_path))
    assert np.array_equal(dm.map, samples.mean(axis=-1).squeeze())


def test_from_mat_file_limits_the_samples(tmp_path: Path) -> None:
    samples = _write_mat(tmp_path)
    dm = DepthMap.from_mat_file("scan.mat", data_folder=str(tmp_path), max_samples=2)
    assert np.array_equal(dm.map, samples[:, :, :, :2].mean(axis=-1).squeeze())


def test_from_mat_file_without_averaging(tmp_path: Path) -> None:
    samples = _write_mat(tmp_path)
    dm = DepthMap.from_mat_file(
        "scan.mat", data_folder=str(tmp_path), average=False, variable_name="other"
    )
    assert np.array_equal(dm.map, samples * 2)


def _write_xyz(folder: Path) -> None:
    rows = [f"{x} {y} 5.0" for x in range(20) for y in range(10)]
    (folder / "cloud.xyz").write_text("\n".join(["header", "header", *rows]), encoding="utf-8")


def test_from_xyz_file_grids_the_points(tmp_path: Path) -> None:
    _write_xyz(tmp_path)
    dm = DepthMap.from_xyz_file("cloud.xyz", data_folder=str(tmp_path))
    assert dm.map.shape == (9, 19)


def test_from_xyz_file_shifts_z(tmp_path: Path) -> None:
    _write_xyz(tmp_path)
    plain = DepthMap.from_xyz_file("cloud.xyz", data_folder=str(tmp_path))
    shifted = DepthMap.from_xyz_file(
        "cloud.xyz", data_folder=str(tmp_path), z_shift=2, rescaled_with=(1.0, 1.0, 1.0)
    )
    assert shifted.map.mean() > plain.map.mean()


def test_heights_use_the_z_resolution() -> None:
    dm = DepthMap(np.array([[2.0, -4.0]]), SensorResolution(1.0, 1.0, 0.5))
    assert dm.heights.tolist() == [[1.0, -2.0]]


def test_save_and_load(tmp_path: Path) -> None:
    original = DepthMap(np.arange(6.0).reshape(2, 3), SensorResolution(1.0, 2.0, 3.0, "cm"))
    path = tmp_path / "map.npz"
    original.save(path)
    loaded = DepthMap.load(path)
    assert np.array_equal(loaded.map, original.map)
    assert loaded.sensor.xyz() == (1.0, 2.0, 3.0)
    assert loaded.sensor.scale == "cm"
