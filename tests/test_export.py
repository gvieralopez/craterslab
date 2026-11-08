import csv
from pathlib import Path

import numpy as np
import pytest

from craterslab.classification import SurfaceType
from craterslab.craters import OBSERVABLES, Surface
from craterslab.export import write_observables_csv, write_profile_csv
from craterslab.profiles import Profile
from craterslab.sensors import DepthMap


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@pytest.fixture
def surfaces(crater: DepthMap) -> dict[str, Surface]:
    return {
        "first": Surface.from_depth_map(crater, SurfaceType.COMPLEX_CRATER),
        "second": Surface.from_depth_map(crater, SurfaceType.SAND_MOUND),
        "third": Surface.from_depth_map(crater),
    }


def test_observables_header(surfaces: dict[str, Surface], tmp_path: Path) -> None:
    path = tmp_path / "observables.csv"
    write_observables_csv(path, surfaces)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["name", "type", *[d.symbol for d in OBSERVABLES]]


def test_observables_rows(surfaces: dict[str, Surface], tmp_path: Path) -> None:
    path = tmp_path / "observables.csv"
    write_observables_csv(path, surfaces)
    rows = _rows(path)
    assert [row["name"] for row in rows] == ["first", "second", "third"]
    assert [row["type"] for row in rows] == ["COMPLEX_CRATER", "SAND_MOUND", "UNKNOWN"]
    expected = surfaces["first"].observables["D"].value
    assert float(rows[0]["D"]) == pytest.approx(expected)


def test_observables_that_do_not_apply_are_empty(
    surfaces: dict[str, Surface], tmp_path: Path
) -> None:
    path = tmp_path / "observables.csv"
    write_observables_csv(path, surfaces)
    rows = _rows(path)
    assert rows[1]["D"] == ""
    assert rows[1]["d_exc"] != ""
    assert all(rows[2][d.symbol] == "" for d in OBSERVABLES)


def test_files_use_plain_line_endings(surfaces: dict[str, Surface], tmp_path: Path) -> None:
    path = tmp_path / "observables.csv"
    write_observables_csv(path, surfaces)
    assert b"\r" not in path.read_bytes()


def test_profile_csv(crater: DepthMap, tmp_path: Path) -> None:
    profile = Profile(crater, (0, 50), (100, 50), total_points=11)
    path = tmp_path / "profile.csv"
    write_profile_csv(path, profile)
    rows = _rows(path)
    assert len(rows) == 11
    assert np.allclose([float(r["distance"]) for r in rows], profile.s)
    assert np.allclose([float(r["height"]) for r in rows], profile.h)
