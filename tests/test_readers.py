from pathlib import Path

import numpy as np
import pytest
import scipy.io

from craterslab.depth_map import SensorResolution
from craterslab.readers import _rescale, read_mat, read_xyz


def _write_mat(folder: Path) -> tuple[Path, np.ndarray]:
    samples = np.arange(16, dtype=float).reshape((2, 2, 1, 4))
    path = folder / "scan.mat"
    scipy.io.savemat(path, {"scan": samples, "other": samples * 2})
    return path, samples


def _write_xyz(folder: Path) -> Path:
    rows = [f"{x} {y} 5.0" for x in range(20) for y in range(10)]
    path = folder / "cloud.xyz"
    path.write_text("\n".join(["header", "header", *rows]), encoding="utf-8")
    return path


def test_mat_averages_the_samples(tmp_path: Path, resolution: SensorResolution) -> None:
    path, samples = _write_mat(tmp_path)
    assert np.array_equal(read_mat(path, resolution).map, samples.mean(axis=-1).squeeze())


def test_mat_limits_the_samples(tmp_path: Path, resolution: SensorResolution) -> None:
    path, samples = _write_mat(tmp_path)
    expected = samples[:, :, :, :2].mean(axis=-1).squeeze()
    assert np.array_equal(read_mat(path, resolution, max_samples=2).map, expected)


def test_mat_without_averaging(tmp_path: Path, resolution: SensorResolution) -> None:
    path, samples = _write_mat(tmp_path)
    depth_map = read_mat(path, resolution, average=False, variable_name="other")
    assert np.array_equal(depth_map.map, samples * 2)


def test_mat_accepts_a_string_path(tmp_path: Path, resolution: SensorResolution) -> None:
    path, samples = _write_mat(tmp_path)
    assert read_mat(str(path), resolution).map.shape == samples.shape[:2]


def test_xyz_grids_the_points(tmp_path: Path, resolution: SensorResolution) -> None:
    assert read_xyz(_write_xyz(tmp_path), resolution).map.shape == (9, 19)


def test_xyz_shifts_z(tmp_path: Path, resolution: SensorResolution) -> None:
    path = _write_xyz(tmp_path)
    plain = read_xyz(path, resolution)
    shifted = read_xyz(path, resolution, z_shift=2)
    assert shifted.map.mean() == pytest.approx(plain.map.mean() + 2)


def test_xyz_rescales_each_axis(tmp_path: Path, resolution: SensorResolution) -> None:
    depth_map = read_xyz(_write_xyz(tmp_path), resolution, rescaled_with=(2.0, 1.0, 1.0))
    assert depth_map.map.shape == (9, 38)


def test_rescale_touches_one_axis_at_a_time() -> None:
    assert _rescale(np.array([[1.0, 2.0, 3.0]]), (2.0, 3.0, 4.0)).tolist() == [[2.0, 6.0, 12.0]]
