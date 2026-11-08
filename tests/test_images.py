from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from craterslab.images import (
    central_regions,
    compute_bounding_box,
    crop_img,
    largest_bounding_box,
    outer_regions,
)

FIXTURES = Path(__file__).parent / "fixtures"
THRESHOLD = 1.5


def test_largest_bounding_box() -> None:
    assert largest_bounding_box(np.zeros((3, 5))) == (0, 0, 5, 3)


def test_crop_img_clips_the_gap_at_the_borders() -> None:
    assert crop_img(np.zeros((10, 10)), 1, 1, 2, 2, gap=5).shape == (8, 8)


@pytest.mark.parametrize(
    ("value", "count"), [(1.5, 0), (1.6, 1), (0.0, 0), (-1.4, 0), (-1.5, 1), (-1.6, 1)]
)
def test_threshold_limits(value: float, count: int) -> None:
    assert outer_regions(np.full((30, 30), value), THRESHOLD)[1] == count


@pytest.mark.parametrize(("size", "count"), [(9, 1), (8, 0)])
def test_opening_removes_regions_that_are_too_small(size: int, count: int) -> None:
    img = np.zeros((40, 40))
    img[15 : 15 + size, 15 : 15 + size] = 5
    assert outer_regions(img, THRESHOLD)[1] == count


def test_opening_keeps_the_size_of_a_region() -> None:
    img = np.zeros((60, 60))
    img[20:32, 22:38] = 5
    assert compute_bounding_box(img, THRESHOLD) == (22, 20, 16, 12)


def test_opening_removes_thin_details_of_a_region() -> None:
    img = np.zeros((70, 70))
    img[25:45, 25:45] = 5
    img[34:36, 45:60] = 5
    assert compute_bounding_box(img, THRESHOLD) == (25, 25, 20, 20)


def test_noise_does_not_make_regions() -> None:
    img = np.zeros((40, 40))
    img[::3, ::3] = 5
    assert outer_regions(img, THRESHOLD)[1] == 0


@pytest.mark.parametrize(("rows", "count"), [(5, 1), (4, 0)])
def test_regions_on_a_border_are_not_worn_away_by_it(rows: int, count: int) -> None:
    img = np.zeros((30, 30))
    img[:rows, :] = 5
    assert outer_regions(img, THRESHOLD)[1] == count


@pytest.mark.parametrize(("size", "count"), [(5, 1), (4, 0)])
def test_regions_in_a_corner(size: int, count: int) -> None:
    img = np.zeros((30, 30))
    img[-size:, -size:] = 5
    assert outer_regions(img, THRESHOLD)[1] == count


def test_regions_touching_diagonally_are_one() -> None:
    img = np.zeros((50, 50))
    img[10:22, 10:22] = 5
    img[22:34, 22:34] = 5
    assert outer_regions(img, THRESHOLD)[1] == 1


def test_regions_with_a_gap_are_two() -> None:
    img = np.zeros((50, 50))
    img[10:22, 10:22] = 5
    img[24:36, 24:36] = 5
    assert outer_regions(img, THRESHOLD)[1] == 2


def _ring(size: int) -> np.ndarray:
    img = np.zeros((size, size))
    img[10 : size - 10, 10 : size - 10] = 5
    img[22 : size - 22, 22 : size - 22] = 0
    return img


def test_a_hole_belongs_to_its_region() -> None:
    regions, count = outer_regions(_ring(80), THRESHOLD)
    assert count == 1
    assert regions[40, 40] == 1


def test_an_island_in_a_hole_is_not_another_region() -> None:
    img = _ring(100)
    img[45:55, 45:55] = -5
    assert outer_regions(img, THRESHOLD)[1] == 1


@pytest.mark.parametrize(("col", "central"), [(85, True), (86, False)])
def test_the_center_distance_limit(col: int, central: bool) -> None:
    regions = np.zeros((100, 100), dtype=int)
    regions[50, col] = 1
    assert central_regions(regions, 1).any() == central


def test_the_image_center_is_compared_with_its_axes_swapped() -> None:
    # Kept from the OpenCV version: (x, y) of a region against (rows, cols) of the image
    regions = np.zeros((40, 100), dtype=int)
    regions[20, 50] = 1
    assert not central_regions(regions, 1).any()


def test_central_regions_keeps_only_the_central_ones() -> None:
    regions = np.zeros((100, 100), dtype=int)
    regions[45:50, 45:50] = 1
    regions[52:58, 52:58] = 2
    regions[0:5, 0:5] = 3
    central = central_regions(regions, 3)
    assert central[47, 47]
    assert central[55, 55]
    assert not central[2, 2]


def test_central_regions_without_regions() -> None:
    assert not central_regions(np.zeros((10, 10), dtype=int), 0).any()


def test_compute_bounding_box_finds_a_central_blob() -> None:
    img = np.zeros((60, 60), dtype=np.float32)
    img[20:40, 25:45] = -5
    assert compute_bounding_box(img, THRESHOLD) == (25, 20, 20, 20)


def test_compute_bounding_box_joins_central_blobs_and_skips_far_ones() -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:50, 40:50] = 5
    img[52:64, 52:64] = -5
    img[2:14, 2:14] = 5
    assert compute_bounding_box(img, THRESHOLD) == (40, 40, 24, 24)


def test_compute_bounding_box_with_only_far_blobs_returns_the_whole_image() -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[2:14, 2:14] = 5
    assert compute_bounding_box(img, THRESHOLD) == (0, 0, 100, 100)


def test_compute_bounding_box_without_regions_returns_the_whole_image() -> None:
    img = np.zeros((8, 6), dtype=np.float32)
    assert compute_bounding_box(img, THRESHOLD) == (0, 0, 6, 8)


def test_compute_bounding_box_returns_plain_integers() -> None:
    img = np.zeros((60, 60))
    img[20:40, 25:45] = -5
    assert all(type(value) is int for value in compute_bounding_box(img, THRESHOLD))


@pytest.mark.parametrize(
    ("name", "threshold", "box"),
    [
        ("fluidized_1", 1.5, (63, 75, 22, 24)),
        ("fluidized_1", 3.0, (64, 77, 20, 20)),
        ("fluidized_10", 1.5, (39, 56, 61, 60)),
        ("fluidized_10", 3.0, (74, 80, 14, 14)),
        ("compacted_1", 1.5, (46, 60, 41, 48)),
        ("compacted_1", 3.0, (47, 65, 36, 38)),
        ("plane_10", 1.5, (0, 0, 157, 157)),
        ("plane_10", 3.0, (0, 0, 157, 157)),
    ],
)
def test_real_depth_maps_keep_the_boxes_opencv_found(
    name: str, threshold: float, box: tuple[int, int, int, int]
) -> None:
    with np.load(FIXTURES / f"{name}.npz") as data:
        depth_map = data["depthmap"]
    assert compute_bounding_box(depth_map, threshold) == box


@pytest.mark.parametrize("seed", range(300))
def test_regions_match_opencv_on_random_surfaces(seed: int) -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    surface = _random_surface(rng)
    threshold = float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]))
    regions, _ = outer_regions(surface, threshold)
    boxes = {
        (s[1].start, s[0].start, s[1].stop - s[1].start, s[0].stop - s[0].start)
        for s in ndimage.find_objects(regions)
    }
    assert boxes == _opencv_boxes(cv2, surface, threshold)


def _random_surface(rng: np.random.Generator) -> np.ndarray:
    height, width = rng.integers(25, 170, 2)
    y, x = np.mgrid[:height, :width]
    surface = np.zeros((height, width), dtype=np.float64)
    for _ in range(rng.integers(0, 8)):
        cx, cy = rng.uniform(-0.1 * width, 1.1 * width), rng.uniform(-0.1 * height, 1.1 * height)
        sx, sy = rng.uniform(2, 0.4 * width + 3), rng.uniform(2, 0.4 * height + 3)
        surface += rng.uniform(-12, 12) * np.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2)
    surface += rng.normal(0, rng.uniform(0, 1.2), surface.shape)
    cx, cy, radius = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(5, 50)
    distances = np.hypot(x - cx, y - cy)
    surface[(distances > radius) & (distances < radius + rng.uniform(6, 25))] = rng.uniform(-9, 9)
    surface[distances < radius * rng.uniform(0.1, 0.6)] = rng.uniform(-9, 9)
    return surface


def _opencv_boxes(cv2: Any, surface: np.ndarray, threshold: float) -> set[tuple[int, ...]]:
    _, low = cv2.threshold(surface, -threshold, 255, cv2.THRESH_BINARY_INV)
    _, high = cv2.threshold(surface, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(low.astype(np.uint8), high.astype(np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(cv2.erode(mask, kernel, iterations=2), kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {tuple(int(v) for v in cv2.boundingRect(contour)) for contour in contours}
