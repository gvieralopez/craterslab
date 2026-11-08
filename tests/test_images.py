import numpy as np

from craterslab.images import compute_bounding_box, crop_img, largest_bounding_box


def test_largest_bounding_box() -> None:
    assert largest_bounding_box(np.zeros((3, 5))) == (0, 0, 5, 3)


def test_crop_img_clips_the_gap_at_the_borders() -> None:
    assert crop_img(np.zeros((10, 10)), 1, 1, 2, 2, gap=5).shape == (8, 8)


def test_compute_bounding_box_finds_a_central_blob() -> None:
    img = np.zeros((60, 60), dtype=np.float32)
    img[20:40, 25:45] = -5
    x, y, w, h = compute_bounding_box(img, threshold=1.5)
    assert (x, y) == (25, 20)
    assert (w, h) == (20, 20)


def test_compute_bounding_box_joins_central_blobs_and_skips_far_ones() -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:50, 40:50] = 5
    img[52:64, 52:64] = -5
    img[2:14, 2:14] = 5
    x, y, w, h = compute_bounding_box(img, threshold=1.5)
    assert (x, y, x + w, y + h) == (40, 40, 64, 64)


def test_compute_bounding_box_with_only_far_blobs_returns_the_whole_image() -> None:
    img = np.zeros((100, 100), dtype=np.float32)
    img[2:14, 2:14] = 5
    assert compute_bounding_box(img, threshold=1.5) == (0, 0, 100, 100)


def test_compute_bounding_box_without_contours_returns_the_whole_image() -> None:
    assert compute_bounding_box(np.zeros((8, 6), dtype=np.float32), threshold=1.5) == (0, 0, 6, 8)
