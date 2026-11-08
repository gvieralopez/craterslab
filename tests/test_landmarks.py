from dataclasses import replace

import numpy as np
import pytest

from craterslab.landmarks import ProfileLandmarks, find_landmarks, wall_lines, wall_slopes
from craterslab.profiles import Profile
from craterslab.sensors import DepthMap, SensorResolution


def _profile(values: np.ndarray, resolution: SensorResolution) -> Profile:
    depth_map = DepthMap(np.tile(values, (3, 1)), resolution)
    return Profile(depth_map, (0, 1), (len(values) - 1, 1), total_points=len(values))


def _piecewise(points: list[tuple[int, float]]) -> np.ndarray:
    x, y = zip(*points, strict=True)
    values: np.ndarray = np.interp(np.arange(x[-1] + 1), x, y)
    return values


def test_landmarks_of_a_crater(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    found = find_landmarks(profile)
    third = len(profile) // 3
    assert found.t1 is not None
    assert found.t2 is not None
    assert found.b1 is not None
    assert found.b2 is not None
    assert found.t1 <= third < 2 * third < found.t2
    assert found.t1 < found.b1
    assert found.b2 < found.t2


def test_landmarks_are_plain_integers(crater: DepthMap) -> None:
    found = find_landmarks(Profile(crater, (0, 50), (100, 50)))
    assert all(type(i) is int for wall in found.walls() for i in wall)


def test_a_flat_profile_has_no_landmarks(resolution: SensorResolution) -> None:
    assert find_landmarks(_profile(np.zeros(90), resolution)) == ProfileLandmarks(
        None, None, None, None
    )


def test_a_profile_without_extrema_has_no_landmarks(resolution: SensorResolution) -> None:
    found = find_landmarks(_profile(np.linspace(0.0, 10.0, 90), resolution))
    assert found == ProfileLandmarks(None, None, None, None)


def test_a_single_bottom_is_used_for_both_walls(resolution: SensorResolution) -> None:
    values = _piecewise([(0, 0.0), (10, 5.0), (35, -5.0), (75, 5.0), (89, 0.0)])
    found = find_landmarks(_profile(values, resolution))
    assert found.b1 == found.b2
    assert found.b1 is not None
    assert abs(found.b1 - 35) <= 2


def test_a_missing_rim_is_reported_as_none(resolution: SensorResolution) -> None:
    values = _piecewise([(0, 0.0), (10, 5.0), (40, -5.0), (89, 8.0)])
    found = find_landmarks(_profile(values, resolution))
    assert found.t2 is None
    assert found.t1 is not None
    assert found.b1 == found.b2


def test_incomplete_landmarks() -> None:
    landmarks = ProfileLandmarks(1, 2, 3, None)
    assert not landmarks.is_complete()
    with pytest.raises(ValueError, match="walls"):
        landmarks.walls()


def test_slopes_of_a_crater(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    left, right = wall_slopes(profile, find_landmarks(profile))
    assert left < 0 < right


def test_slopes_are_plain_floats(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    assert all(type(s) is float for s in wall_slopes(profile, find_landmarks(profile)))


def test_landmarks_can_be_adjusted(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    found = find_landmarks(profile)
    assert found.t1 is not None
    moved = replace(found, t1=found.t1 + 50)
    assert wall_slopes(profile, moved) != wall_slopes(profile, found)


def test_a_line_needs_two_points(resolution: SensorResolution) -> None:
    profile = _profile(np.arange(30.0), resolution)
    with pytest.raises(ValueError, match="Not enough points"):
        wall_lines(profile, ProfileLandmarks(5, 5, 10, 20))


def test_line_endpoints_can_be_given_in_any_order(crater: DepthMap) -> None:
    profile = Profile(crater, (0, 50), (100, 50))
    forward = wall_lines(profile, ProfileLandmarks(100, 300, 700, 900))
    backward = wall_lines(profile, ProfileLandmarks(300, 100, 900, 700))
    assert np.array_equal(forward[0], backward[0])
    assert np.array_equal(forward[1], backward[1])
