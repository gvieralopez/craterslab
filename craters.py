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
        image_depth: float
    ):
        self.image_before = image_before
        self.image_after = image_after
        self.image_resolution = image_resolution
        self.image_depth = image_depth
        self.scale = (image_resolution, image_resolution, image_depth)

        self.image_crater = self._compute_crater_image()

    def _compute_crater_image(self) -> np.ndarray:
        # TODO: Crop resulting image to the effective crater region
        return self.image_before - self.image_after

    def _create_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.arange(0, self.image_crater.shape[0] * self.scale[0], self.scale[0])
        y = np.arange(0, self.image_crater.shape[1] * self.scale[1], self.scale[1])
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

    def plot_3D(
        self,
        title: str,
        preview_scale: tuple[float, float, float],
    ):

        # Create a mesh grid
        X, Y, Z = self._create_mesh()

        # Create the plot
        self._plot_3D(X, Y, Z, title, preview_scale)

    def plot_profile(
        self,
        title: str,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        total_points: int = 1000,
    ):

        # Extract the line
        # Make a line with "num" points...
        x0, y0 = start_point  # These are in _pixel_ coordinates!!
        x1, y1 = end_point
        x, y = np.linspace(x0, x1, total_points), np.linspace(y0, y1, total_points)

        # Extract the values along the line, using cubic interpolation
        zi = scipy.ndimage.map_coordinates(self.image_crater, np.vstack((x, y))).flatten()
        zi = list(self.image_depth * zi)
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = np.linspace(0, self.image_resolution * np.sqrt(dx**2 + dy**2), len(zi))

        # -- Plot...
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        axes[0].imshow(self.image_crater)
        axes[0].set_xlabel("X [px]", fontweight="bold")
        axes[0].set_ylabel("Y [px]", fontweight="bold")
        axes[0].set_title("Top View of the crater", fontweight="bold")
        axes[0].plot([x0, x1], [y0, y1], "ro-")
        axes[0].axis("image")

        # axes[1].plot(distance, zi)
        axes[1].plot(distance, zi)
        axes[1].set_title("Profile View of the crater", fontweight="bold")
        axes[1].set_ylabel("Depth [mm]", fontweight="bold")
        axes[1].set_xlabel("Distance [mm]", fontweight="bold")

        plt.show()
