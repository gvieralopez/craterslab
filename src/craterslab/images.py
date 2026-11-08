import cv2
import numpy as np
from scipy.spatial import distance


def largest_bounding_box(img: np.ndarray) -> tuple[int, int, int, int]:
    h, w = img.shape
    return 0, 0, w, h


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


def select_best_countour(img: np.ndarray, contours: np.ndarray) -> np.ndarray:
    # find center of image and draw it (blue circle)
    image_center = np.asarray(img.shape) / 2
    image_center = tuple(image_center.astype("int32"))

    max_distance = distance.euclidean(image_center, img.shape)
    valid_contours = []
    for contour in contours:
        # find center of each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = (center_X, center_Y)

            # calculate distance to image_center
            distances_to_center = (
                distance.euclidean(image_center, contour_center)
            ) / max_distance

            # save to a list of dictionaries
            valid_contours.append(
                {
                    "contour": contour,
                    "center": contour_center,
                    "distance_to_center": distances_to_center,
                }
            )

    central_contours = [
        c["contour"] for c in valid_contours if c["distance_to_center"] < 0.5
    ]

    # Concatenate all contours
    return np.concatenate(central_contours)


def compute_bounding_box(
    img: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    """
    Compute the bounding box of an image for the region in which pixels are
    |p| > threshold
    """
    # Threshold the image
    _, thresh1 = cv2.threshold(img, -threshold, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    thresh1, thresh2 = thresh1.astype(np.uint8), thresh2.astype(np.uint8)
    thresh = cv2.bitwise_or(thresh1, thresh2)

    # Open the image
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find the contours of the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if len(contours):
        if len(contours) == 1:
            best_countour = contours[0]
        best_countour = select_best_countour(img, contours)

        # Find the bounding rectangle of the largest contour
        return cv2.boundingRect(best_countour)
    # Return the whole image bounding rectangle
    return largest_bounding_box(img)
