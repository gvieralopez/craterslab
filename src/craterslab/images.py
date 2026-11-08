import numpy as np
from scipy import ndimage
from scipy.spatial import distance

MAX_CENTER_DISTANCE = 0.5
OPENING_SIZE = 5
OPENING_ITERATIONS = 2
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def crop_img(img: np.ndarray, x: int, y: int, w: int, h: int, gap: int) -> np.ndarray:
    """
    Crop an image given a bounding box (x, y, w, h) and a gap value for padding
    """
    y_max, x_max = img.shape
    y0 = max(0, y - gap)
    ym = min(y_max, y + h + gap)
    x0 = max(0, x - gap)
    xm = min(x_max, x + w + gap)
    return img[y0:ym, x0:xm]


def compute_bounding_box(img: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """
    Compute the bounding box of the image region whose pixels have an absolute
    value above the threshold
    """
    rows, cols = np.nonzero(central_regions(*outer_regions(img, threshold)))
    if rows.size == 0:
        # Nothing stands out near the center, so keep the whole image
        return largest_bounding_box(img)

    x, y = int(cols.min()), int(rows.min())
    return x, y, int(cols.max()) - x + 1, int(rows.max()) - y + 1


def outer_regions(img: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    mask = (img <= -threshold) | (img > threshold)

    # Pixels beyond the border count as set while eroding, so that regions
    # touching the border are not worn away by it
    kernel = np.ones((OPENING_SIZE, OPENING_SIZE), dtype=bool)
    mask = ndimage.binary_erosion(mask, kernel, iterations=OPENING_ITERATIONS, border_value=1)
    mask = ndimage.binary_dilation(mask, kernel, iterations=OPENING_ITERATIONS)

    # Holes, and anything inside them, belong to the region around them
    regions, count = ndimage.label(ndimage.binary_fill_holes(mask), structure=EIGHT_CONNECTED)
    return regions, int(count)


def central_regions(regions: np.ndarray, count: int) -> np.ndarray:
    height, width = regions.shape
    image_center = (width // 2, height // 2)
    max_distance = distance.euclidean(image_center, (width, height))
    central = np.zeros(regions.shape, dtype=bool)
    for label in range(1, count + 1):
        region = regions == label
        row, col = ndimage.center_of_mass(region)
        distance_to_center = distance.euclidean(image_center, (int(col), int(row))) / max_distance
        if distance_to_center < MAX_CENTER_DISTANCE:
            central |= region
    return central


def largest_bounding_box(img: np.ndarray) -> tuple[int, int, int, int]:
    h, w = img.shape
    return 0, 0, w, h
