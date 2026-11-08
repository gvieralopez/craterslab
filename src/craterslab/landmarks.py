from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from craterslab.profiles import Profile


@dataclass(frozen=True, slots=True)
class ProfileLandmarks:
    t1: int | None
    b1: int | None
    b2: int | None
    t2: int | None

    def walls(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.t1 is None or self.b1 is None or self.b2 is None or self.t2 is None:
            raise ValueError("The walls need all the landmarks of a crater")
        return (self.t1, self.b1), (self.b2, self.t2)

    def is_complete(self) -> bool:
        return None not in (self.t1, self.b1, self.b2, self.t2)


def find_landmarks(profile: Profile) -> ProfileLandmarks:
    extrema = np.where(np.diff(np.sign(np.gradient(profile.h, profile.s))))[0]
    lowest_first = [int(i) for i in extrema[np.argsort(profile.h[extrema])]]
    highest_first = lowest_first[::-1]
    length, third = len(profile), len(profile) // 3

    t1 = _first(i for i in highest_first if i <= third)
    t2 = _first(i for i in highest_first if i > 2 * third)
    b1 = _first(i for i in lowest_first if i <= length // 2 and (t1 is None or i > t1))
    b2 = _first(i for i in lowest_first if i > length // 2 and t2 is not None and i < t2)
    if b1 is None or b2 is None:
        b1 = b2 = b1 if b2 is None else b2
    return ProfileLandmarks(t1, b1, b2, t2)


def wall_lines(profile: Profile, landmarks: ProfileLandmarks) -> tuple[np.ndarray, np.ndarray]:
    first, second = landmarks.walls()
    return _fit_line(profile, *first), _fit_line(profile, *second)


def wall_slopes(profile: Profile, landmarks: ProfileLandmarks) -> tuple[float, float]:
    first, second = wall_lines(profile, landmarks)
    return float(first[0]), float(second[0])


def _first(indices: Iterator[int]) -> int | None:
    return next(indices, None)


def _fit_line(profile: Profile, i: int, j: int) -> np.ndarray:
    start, end = sorted((i, j))
    distances, heights = profile.s[start:end], profile.h[start:end]
    if distances.size <= 1:
        raise ValueError("Not enough points for slope calculation.")
    return np.polyfit(distances, heights, 1)
