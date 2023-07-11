import logging
import math

import numpy as np


class Crater:
    """
    Computes and stores all the crater observables given two depth maps from
    before and after the impact
    """

    def __init__(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        image_resolution: float,
        image_depth: float,
        ellipse_points: int = 20,
    ):
        self.image_resolution = image_resolution
        self.image_depth = image_depth

        self._crater_image(image_before, image_after)
        ellipse = EllipticalModel(self.img, ellipse_points)
        self.ellipse = ellipse if ellipse.is_ok else None
        self._set_default_profile()

        if self.is_valid:
            self._compute_observables()

    def _set_profile(self, p1, p2) -> bool:
        try:
            self._profile = Profile(
                self.img,
                start_point=p1,
                end_point=p2,
                xy_resolution=self.image_resolution,
                z_resolution=self.image_depth,
            )
            return True

        except ValueError:
            return False

    def _set_default_profile(self):
        if self.ellipse is not None:
            p1, p2 = self.ellipse.max_profile_bounds()
            if self._set_profile(p1, p2):
                return
        p1, p2 = (0, 0), self.img.shape
        self._set_profile(p1, p2)
        logging.warning(
            "Failed to set default profile using the elliptical model."
            "Using default profile as y=x"
        )

    def __repr__(self):
        if self.is_valid:
            return self._observables_summary()
        logging.error("Data does not seems to represent a valid crater")
        return ""

    @property
    def is_valid(self) -> bool:
        """
        Return a boolean to indicate whether the data is likely to be from a
        valid crater
        """
        return self.ellipse is not None

    @property
    def scale(self):
        """
        Provides a tuple indicating the sensor resolution on each axis
        """
        return (self.image_resolution, self.image_resolution, self.image_depth)

    @property
    def data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provides de-normalized crater data (i.e, data is presented on actual
        scale computed using the characteristics of the sensor)
        """
        y = np.linspace(0, self.img.shape[0] * self.scale[0], self.img.shape[0])
        x = np.linspace(0, self.img.shape[1] * self.scale[1], self.img.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = self.img * self.scale[2]
        return X, Y, Z

    def _compute_observables(self):
        self.da = np.min(self.img) * self.image_depth
        self.D = 2 * abs(self.ellipse.a) * self.image_resolution

    def _observables_summary(self):
        return f"""
        Crater observables computed:
        ----------------------------

        - Max depth: {self.da:.2f} mm 
        - Diameter: {self.D:.2f} mm 
        """
