import cv2
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Crater:
    def __init__(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        image_resolution: float,
        image_depth: float,
    ):
        self.image_before = image_before
        self.image_after = image_after
        self.image_resolution = image_resolution
        self.image_depth = image_depth
        self.scale = (image_resolution, image_resolution, image_depth)

        self.image_crater = self._compute_crater_image()
        self.profile = None
        self.profile_distance = None

    def _compute_crater_image(
        self, diff_threshold: int = 3, padding: int = 20
    ) -> np.ndarray:
        # Compute the difference
        diff = self.image_before - self.image_after
        h_max, w_max = diff.shape

        # Threshold the image
        _, thresh = cv2.threshold(
            diff, -diff_threshold, diff_threshold, cv2.THRESH_BINARY_INV
        )
        thresh = thresh.astype(np.uint8)

        # Find the contours of the binary image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the original image using the coordinates of the bounding rectangle
        y0 = max(0, y - padding)
        ym = min(h_max, y + h + padding)
        x0 = max(0, x - padding)
        xm = max(w_max, x + w + padding)
        return diff[y0:ym, x0:xm]

    def _create_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.arange(0, self.image_crater.shape[0] * self.scale[0], self.scale[0])
        x = np.arange(0, self.image_crater.shape[1] * self.scale[1], self.scale[1])
        X, Y = np.meshgrid(x, y)
        # TODO: Evaluate wether to to this here or elsewhere
        Z = self.image_crater * self.scale[2]
        return X, Y, Z

    def _plot_3D(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        title: str,
        preview_scale: tuple[float, float, float],
    ):

        # Set up the plot
        ax = plt.figure().gca(projection="3d")
        surface = ax.plot_surface(X, Y, Z, cmap="viridis")

        # Preserve aspect ratio
        ax.set_box_aspect(
            [
                preview_scale[0] * np.ptp(X),
                preview_scale[1] * np.ptp(Y),
                preview_scale[2] * np.ptp(Z),
            ]
        )

        # Write labels, title and color bar
        ax.set_xlabel("X[mm]", fontweight="bold")
        ax.set_ylabel("Y[mm]", fontweight="bold")
        ax.set_zlabel("Z[mm]", fontweight="bold")
        plt.title(title)
        plt.colorbar(surface, shrink=0.5, aspect=10, orientation="horizontal", pad=0.2)

        # Show result
        plt.show()

    def _compute_extremes(self):
        dydx = np.gradient(self.profile, self.profile_distance)
        extrema_indices = np.where(np.diff(np.sign(dydx)))[0]

        print(self.profile)

        # Get the indices that would sort y in ascending order
        sorted_indices = np.argsort(self.profile[extrema_indices])
        smallest_indices = sorted_indices[:2]
        largest_indices = sorted_indices[-2:]

        # Select the corresponding elements from the extrema indices
        selected_subindices = np.concatenate([smallest_indices, largest_indices])
        selected_indices = extrema_indices[selected_subindices]
        self.t1, self.b1, self.b2, self.t2 = np.sort(selected_indices)

    def _compute_slopes(self):
        p1 = np.polyfit(
            self.profile_distance[self.t1 : self.b1], self.profile[self.t1 : self.b1], 1
        )
        self.slope1 = np.poly1d(p1)
        p2 = np.polyfit(
            self.profile_distance[self.b2 : self.t2], self.profile[self.b2 : self.t2], 1
        )
        self.slope2 = np.poly1d(p2)

    def plot_3D(
        self,
        title: str,
        preview_scale: tuple[float, float, float],
    ):

        # Create a mesh grid
        X, Y, Z = self._create_mesh()

        # Create the plot
        self._plot_3D(X, Y, Z, title, preview_scale)

    def set_profile(
        self,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        total_points: int = 1000,
    ):
        # Extract the line
        x0, y0 = start_point  # These are in _pixel_ coordinates!!
        x1, y1 = end_point
        x, y = np.linspace(x0, x1, total_points), np.linspace(y0, y1, total_points)

        # Extract the values along the line, using cubic interpolation
        zi = scipy.ndimage.map_coordinates(
            self.image_crater, np.vstack((x, y))
        ).flatten()
        self.profile = self.image_depth * zi

        dx, dy = x1 - x0, y1 - y0

        self.profile_distance = np.linspace(
            0, self.image_resolution * np.sqrt(dx**2 + dy**2), len(self.profile)
        )

        self.profile_bounds = [[x0, x1], [y0, y1]]
        self._compute_extremes()
        self._compute_slopes()

    def plot_profile(self, title: str):
        if self.profile is not None:
            # -- Plot...
            fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
            axes[0].imshow(self.image_crater)
            axes[0].set_xlabel("X [px]", fontweight="bold")
            axes[0].set_ylabel("Y [px]", fontweight="bold")
            axes[0].set_title("Top View of the crater", fontweight="bold")
            axes[0].plot(*self.profile_bounds, "ro-")
            axes[0].axis("image")

            axes[1].plot(self.profile_distance, self.profile)
            axes[1].set_title("Profile View of the crater", fontweight="bold")
            axes[1].plot(
                [self.profile_distance[self.t1], self.profile_distance[self.b1]],
                [
                    self.slope1(self.profile_distance[self.t1]),
                    self.slope1(self.profile_distance[self.b1]),
                ],
                color="blue",
                linestyle="--",
            )
            axes[1].plot(
                [self.profile_distance[self.t2], self.profile_distance[self.b2]],
                [
                    self.slope2(self.profile_distance[self.t2]),
                    self.slope2(self.profile_distance[self.b2]),
                ],
                color="blue",
                linestyle="--",
            )
            axes[1].set_ylabel("Depth [mm]", fontweight="bold")
            axes[1].set_xlabel("Distance [mm]", fontweight="bold") 
            selected_indices = np.array([self.t1, self.b1, self.b2, self.t2])
            axes[1].scatter(self.profile_distance[selected_indices], self.profile[selected_indices])

            plt.show()
        else:
            print(
                "No profile has been set yet. You need to call ´Crater.set_profile´ first"
            )
