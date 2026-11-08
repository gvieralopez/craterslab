import math

import numpy as np
import pytest

from craterslab.profiles import Profile
from craterslab.sensors import DepthMap, SensorResolution


@pytest.fixture
def ramp() -> DepthMap:
    y, x = np.mgrid[:50, :50]
    return DepthMap((x + y).astype(float), SensorResolution(2.0, 2.0, 0.5, "cm"))


def test_rejects_different_x_and_y_resolution() -> None:
    dm = DepthMap(np.zeros((5, 5)), SensorResolution(1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="resolution"):
        Profile(dm, (0, 0), (4, 4))


def test_length_and_scaled_arrays(ramp: DepthMap) -> None:
    p = Profile(ramp, (0, 0), (30, 40), total_points=11)
    assert len(p) == 11
    assert p.s[-1] == pytest.approx(100.0)
    assert p.h[-1] == pytest.approx(35.0, abs=0.01)


def test_bounds(ramp: DepthMap) -> None:
    p = Profile(ramp, (1, 2), (3, 4))
    assert p.x_bounds_px == (1, 3)
    assert p.y_bounds_px == (2, 4)
    assert p.x_bounds == (2.0, 6.0)
    assert p.y_bounds == (4.0, 8.0)
    assert p.z_bounds[0] < p.z_bounds[1]


def test_angle(ramp: DepthMap) -> None:
    assert Profile(ramp, (0, 0), (10, 10)).ang == pytest.approx(math.pi / 4)


def test_index_to_coordinates(ramp: DepthMap) -> None:
    p = Profile(ramp, (0, 0), (40, 0), total_points=41)
    assert p.index2xyz_pix(20) == (20, 0, 20)
    assert p.index2xyz(20) == pytest.approx((40.0, 0.0, 10.0), abs=0.01)


def test_extrema_of_a_crater(crater: DepthMap) -> None:
    p = Profile(crater, (0, 50), (100, 50))
    p.compute_extrema()
    third = len(p) // 3
    assert p.t1 <= third < 2 * third < p.t2
    assert p.t1 < p.b1 < p.b2 < p.t2 or p.b1 == p.b2


def test_slopes_of_a_crater(crater: DepthMap) -> None:
    p = Profile(crater, (0, 50), (100, 50))
    left, right = p.slopes()
    assert left < 0 < right


def test_landmarks_can_be_overridden(crater: DepthMap) -> None:
    p = Profile(crater, (0, 50), (100, 50))
    p.set_landmark_indices(t1=10, b1=20, t2=90, b2=80)
    assert (p.t1, p.b1, p.t2, p.b2) == (10, 20, 90, 80)


def test_line_rejects_missing_indices(crater: DepthMap) -> None:
    with pytest.raises(ValueError, match="indices"):
        Profile(crater, (0, 50), (100, 50))._compute_line(None, 5)  # type: ignore[arg-type]


def test_only_one_bottom_found_is_used_for_both(resolution: SensorResolution) -> None:
    h = np.zeros((3, 40))
    h[:, :] = [abs(i - 14) for i in range(40)]
    h[:, 20:] = np.arange(20, 0, -1) - 8.0
    p = Profile(DepthMap(h, resolution), (0, 1), (39, 1), total_points=1000)
    p.compute_extrema()
    assert p.b1 == p.b2 or -1 not in (p.b1, p.b2)
