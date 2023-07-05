import math
import logging

import cv2
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from pycraters.elipse import fit_elipse
from collections.abc import Callable
from scipy.spatial import distance

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

def select_best_countour(img, contours):

    #find center of image and draw it (blue circle)
    image_center = np.asarray(img.shape) / 2
    image_center = tuple(image_center.astype('int32'))
    
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
            distances_to_center = (distance.euclidean(image_center, contour_center))/max_distance
        
            # save to a list of dictionaries
            valid_contours.append({
                'contour': contour, 
                'center': contour_center, 
                'distance_to_center': distances_to_center})
    
    central_contours = [c['contour'] for c in valid_contours if c['distance_to_center'] < .5]
    
    # Concatenate all contours
    return np.concatenate(central_contours)


def compute_bounding_box(img: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
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
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imshow('dil', thresh)

    # Find the contours of the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if len(contours):
        if len(contours) == 1:
            best_countour = contours[0]
        best_countour = select_best_countour(img, contours)

        # Find the bounding rectangle of the largest contour
        return cv2.boundingRect(best_countour)    


class Profile:
    """
    Computes the profile of a crater through a segment of a depth map:

        t1                       t2     
           /\                  /\       
    ______/  \      tc        /  \______
              \      /\      /          
               \    /  \    /           
                \__/    \__/            
             b1             b2          
    """

    def __init__(
        self,
        img: np.ndarray,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        total_points: int = 1000,
        xy_resolution: float = 1.,
        z_resolution: float = 1.,
    ):
        self.img = img
        self.x0, self.y0 = start_point
        self.xf, self.yf = end_point
        self.total_points = total_points
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution
        self._compute_profile()
    
    def __len__(self):
        """ 
        Provide the size of the profile
        """
        return len(self._h)
       

    @property
    def x_pixel_bounds(self):
        """
        Provide a tuple indicating the pixel bounds of the profile in the x axis
        """
        return (self.x0, self.xf)
    
    @property
    def y_pixel_bounds(self):
        """
        Provide a tuple indicating the pixel bounds of the profile in the y axis
        """
        return (self.y0, self.yf)
    
    @property
    def x_bounds(self):
        """
        Provide a tuple indicating the pixel bounds of the profile in the x axis
        """
        return (self.x0 * self.xy_resolution, self.xf * self.xy_resolution)
    
    @property
    def y_bounds(self):
        """
        Provide a tuple indicating the pixel bounds of the profile in the y axis
        """
        return (self.y0 * self.xy_resolution, self.yf * self.xy_resolution)
    
    @property
    def ang(self):
        """
        Provide the angle of the profile
        """
        return math.atan2(self.yf - self.y0, self.xf - self.x0)
    
    @property
    def h(self):
        """
        Provide the height profile of the crater
        """
        return self.z_resolution * self._h

    @property
    def s(self):
        """
        Provide the displacement array on which the height profile was computed
        """
        return self.xy_resolution * self._s
    
    @property
    def key_indexes(self):
        """ 
        Provide a tuple with all the key indexes for: t1, b1, tc, b2, t2
        """
        return self.t1, self.b1, self.tc, self.b2, self.t2
    
    @property
    def walls(self):
        """ 
        Provide the points that bound each crater wall in the profile. 
        Result is returned as: [(x11, x12), (z11, z12)], [(x21, x22), (z21, z22)].
        So, first the bounds for the left wall and then the bounds for the right
        wall.
        """
        left_wall = self._compute_wall_bounds(1)    
        right_wall = self._compute_wall_bounds(2) 
        return left_wall, right_wall
    
    def _compute_profile(self):
        """
        Computes all the necessary attributes for a profile
        """
        self._compute_height()
        self._compute_displacement()
        self._compute_key_indexes()
        self._compute_slopes()

    def _allocate_points(self) -> tuple[np.array, np.array]:
        """
        Reserve memory for all the points that compose the profile

        """
        x = np.linspace(self.x0, self.xf, self.total_points)
        y = np.linspace(self.y0, self.yf, self.total_points)
        return x, y

    def _compute_height(self):
        """
        Compute height (h) for each profile point according to the depth-map img

        """
        x, y = self._allocate_points()
        zi = scipy.ndimage.map_coordinates(self.img, np.vstack((x, y)))
        self._h = zi.flatten()

    def _compute_displacement(self):
        """
        Compute the displacement (s) on each profile point

        """
        dx, dy = self.xf - self.x0, self.yf - self.y0
        max_displacement = np.sqrt(dx**2 + dy**2)
        self._s = np.linspace(0, max_displacement, self.total_points)

    def _reset_key_indexes(self):
        """ 
        Set the default value for the key indexes
        """
        self.t1, self.b1, self.tc, self.b2, self.t2 = None, None, None, None, None
    
    def _compute_sorted_extreme_indexes(self):
        """ 
        Compute the indices of the array where the array has an extreme (max or min)
        and sort them according to the value of the array in that index in ascending 
        order.
        """
        dhds = np.gradient(self._h, self._s)
        extrema_indices = np.where(np.diff(np.sign(dhds)))[0]
        return extrema_indices[np.argsort(self.h[extrema_indices])]

    def _compute_key_indexes(self):
        """
        Compute the 5 key indexes of the array: t1, b1, tc, b2, t2. 
        See the class main docstring for reference.
        """
        self._reset_key_indexes()
        sorted_indices = self._compute_sorted_extreme_indexes()
        self._compute_max_indexes(sorted_indices)
        self._compute_min_indexes(sorted_indices)

    def _compute_min_indexes(self, sorted_indices):
        """
        Compute b1 and b2
        """
        self.b1 = next(filter(lambda i: i <= len(self) // 2, sorted_indices), self.b1)
        self.b2 = next(filter(lambda i: i > len(self) // 2, sorted_indices), self.b2)
        if None in (self.b1, self.b2):
            raise ValueError("Could not compute minimum indexes.")       

    def _compute_max_indexes(self, sorted_indices):
        """
        Compute t1, tc and t2
        """
        indices = sorted_indices[::-1]
        third = len(self) // 3
        self.t1 = next(filter(lambda i: i <= third, indices), self.t1)
        self.tc = next(filter(lambda i: third  < i <= 2 * third, indices), self.tc)
        self.t2 = next(filter(lambda i: i > 2 * third, indices), self.t2)
        if None in (self.t1, self.t2, self.tc):
            raise ValueError("Could not compute maximum indexes.")

    def _compute_slope(self, i: int, j: int):
        """
        Compute the slope of the line that better fits the profile from i to j
        """
        if None in (i, j):
            raise ValueError("Invalid indices for slope calculation.")
        i, j = (i, j) if i < j else (j, i)
        x_vals, y_vals = self.s[i : j], self.h[i : j]
        if x_vals.size > 1:
            try:
                p1 = np.polyfit(x_vals, y_vals , 1)
                return np.poly1d(p1)
            except SystemError:
                logging.error('Error computing slopes')

    def _compute_slopes(self):
        """
        Compute the slope of crater walls
        """
        self.slope1 = self._compute_slope(self.t1, self.b1)
        self.slope2 = self._compute_slope(self.b2, self.t2)
    
    def _compute_wall_bounds(self, wall: int = 1):
        """ 
        Compute the bounds of a given wall given the id (1 or 2) to refer to 
        the left or right wall respectively.
        """
        assert wall in (1, 2)
        if wall == 1:
            return self._single_wall_bound(self.t1, self.b1, self.slope1)
        return self._single_wall_bound(self.t2, self.b2, self.slope2)
    
    def _single_wall_bound(self, i: int, j: int, model: Callable):
        """ 
        Return the wall bounds given the indexes of the bounds (i, j) and the 
        linear function that interpolates them.
        """
        x = [self.s[i], self.s[j]]
        if model:
            z = [model(v) for v in x]
        else:
            z = x
            logging.error('Slopes are invalid')
        return x, z


class EllipticalModel:
    """
    Computes an ellipse that fits the crater rims on a depth map
    """

    def __init__(self, img: np.ndarray, points: int):
        self.img = img
        self.points = points
        self.MIN_REQ_POINTS = 4
        self._compute_landmark_points(points)
        if self._validate_landmark_points():
            x = np.array([p[0] for p in self.landmarks])
            y = np.array([p[1] for p in self.landmarks])
            ymax, xmax = self.img.shape
            xrange, yrange = (0, xmax), (0, ymax)
            self.a, self.b, self.cx, self.cy, self.theta = fit_elipse(x, y, xrange, yrange)
            self.is_ok = True
        else:
            self.is_ok = False

    def _compute_landmark_points(self, points):
        assert points % 2 == 0
        bounds = self._compute_bounds(points)
        self.landmarks = self._compute_landmarks(bounds)

    def _validate_landmark_points(self):
        if len(self.landmarks) < self.points:
            logging.warning(f"Only {len(self.landmarks)} landmark were computed " 
                            f"(instead of {self.points})")
            if len(self.landmarks) < self.MIN_REQ_POINTS:                
                logging.error(f"{len(self.landmarks)} landmarks are not sufficient for "
                              "fitting the elliptical model")
                return False
            else:            
                logging.warning(f"Trying to fit the elliptical model with {len(self.landmarks)} "
                              "landmarks")
        return True
        
    def _compute_bounds(self, points: int):
        M, N = self.img.shape
        center_y, center_x = M // 2, N // 2
        angle_diff = 2 * np.pi / points
        return [
            self._compute_profile_bounds(i * angle_diff, center_x, center_y)
            for i in range(math.ceil(points / 2))
        ]

    def _compute_max_radius(self, angle, center_x, center_y) -> float:
        cos_angle = np.abs(np.cos(angle))
        sin_angle = np.abs(np.sin(angle))
        if cos_angle == 0:
            return center_y / sin_angle
        elif sin_angle == 0:
            return center_x / cos_angle
        return min(center_y / sin_angle, center_x / cos_angle)


    def _compute_profile_bounds(self, angle, center_x, center_y) -> list:
        max_radius = self._compute_max_radius(angle, center_x, center_y)
        x0 = int(round(center_x + max_radius * np.cos(angle)))
        y0 = int(round(center_y + max_radius * np.sin(angle)))
        xf = int(round(center_x + max_radius * np.cos(angle + np.pi)))
        yf = int(round(center_y + max_radius * np.sin(angle + np.pi)))
        return [(x0, y0), (xf, yf)]

    def _compute_landmarks(self, bounds: list):
        landmarks = []
        for bound in bounds:
            try:
                l1, l2 = self._compute_complementary_landmarks(bound)
                landmarks += [l1, l2]
            except:
                logging.error('Cannot compute a landmark pair')
        return landmarks

    def _compute_complementary_landmarks(self, bound):
        try:
            p = Profile(self.img, bound[0], bound[1])
            l1 = self._compute_single_landmark(p, p.t1)
            l2 = self._compute_single_landmark(p, p.t2)
            return l1, l2
        except ValueError:
            raise ValueError("Could not create a profile for those bounds")
    
    def _compute_single_landmark(self, profile: Profile, index: int):
        d = profile._s[index]
        x1 = profile.x0 + d * math.cos(profile.ang)
        y1 = profile.y0 + d * math.sin(profile.ang)
        return x1, y1
    
    def _scale_ellipse(self, cx, cy, da, db, scale):
        return cx * scale, cy * scale, da * scale, db * scale
    
    def ellipse_patch(self, scale: float = 1., color=None):
        cx, cy = self.cx, self.cy
        da, db = 2 * self.a, 2 * self.b
        theta_degree = self.theta * 180 / np.pi
        if scale != 1.:
            cx, cy, da, db = self._scale_ellipse(cx, cy, da, db, scale)
        return Ellipse((cx, cy), da, db, theta_degree, fill=False, color=color)
    
    def max_profile_bounds(self):
        return self._compute_profile_bounds(self.theta, self.cx, self.cy)


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
            self._profile = Profile(self.img,
                                start_point=p1, 
                                end_point=p2, 
                                xy_resolution=self.image_resolution, 
                                z_resolution=self.image_depth)
            return True
                         
        except ValueError:
            return False

    def _set_default_profile(self):
        if self.ellipse is not None:
            p1, p2 = self.ellipse.max_profile_bounds()
            if self._set_profile(p1, p2):
                return  
        p1, p2 = (0,0), self.img.shape
        self._set_profile(p1, p2)
        logging.warning("Failed to set default profile using the elliptical model."
                        "Using default profile as y=x")

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

    def plot_3D(
        self,
        title: str = 'Crater view in 3D',
        preview_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """ 
        Create a 3D plot of the crater considering the sensor resolution. 
        Preview scale can be passed to enlarge or shrink specific dimensions.
        """

        # Create the plot with de-normalized data
        self._plot_3D(*self.data, title, preview_scale)


    def plot_profile(self, title: str, profile: Profile | None = None, only_profile: bool = False):
        """ 
        Create a profile plot of the crater considering the sensor resolution. 
        If no specific profile is passed, the plot will be shown on the largest
        profile computed used the elliptical approximation.
        """
        if profile is None:
            profile = self._profile
        if only_profile:
            self._plot_profile_only(title, profile)
        else:
            self._plot_profile(title, profile)
    
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

    def _crater_image(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        diff_threshold: int = 1,
        padding: int = 10,
    ) -> np.ndarray:

        diff = image_before - image_after
        bb = compute_bounding_box(diff, threshold=diff_threshold)
        self.img = crop_img(diff, *bb, padding) if bb is not None else diff

    def _plot_3D(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        title: str,
        preview_scale: tuple[float, float, float],
    ):

         # Set up the plot
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        surface = ax.plot_surface(X, Y, Z, cmap="viridis")

        # Preserve aspect ratio
        aspect = [
            preview_scale[0] * np.ptp(X), 
            preview_scale[1] * np.ptp(Y),
            preview_scale[2] * np.ptp(Z),
        ]
        ax.set_box_aspect(aspect)


        # if self.is_valid:
        #     p = self.ellipse.ellipse_patch(scale=self.image_resolution, color="red")
        #     ax.add_patch(p)
        #     art3d.pathpatch_2d_to_3d(p, z=max(self._profile.h), zdir='z') 

        # Write labels, title and color bar
        ax.set_xlabel("X[mm]", fontweight="bold", fontsize=13, labelpad=10)
        ax.set_ylabel("Y[mm]", fontweight="bold", fontsize=13, labelpad=10)
        ax.set_zlabel("Z[mm]", fontweight="bold", fontsize=13, labelpad=10)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='z', labelsize=13)
        plt.title(title)
        plt.colorbar(surface, shrink=0.5, aspect=10, orientation="vertical", pad=0.2)
        plt.show()


    def _plot_profile_only(self, title: str, profile: Profile):

        fig, ax = plt.subplots()  #create figure and axes
        ax.plot(profile.s, profile.h)
        for x_bounds, z_bounds in profile.walls:
            ax.plot(x_bounds, z_bounds, color="blue", linestyle="--")  
        ax.set_ylabel("Depth [mm]", fontweight="bold", labelpad=13, fontsize=30)
        ax.set_xlabel("Distance [mm]", fontweight="bold", labelpad=13, fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        selected_indices = np.array(profile.key_indexes)
        ax.scatter(profile.s[selected_indices], profile.h[selected_indices])
        ax.grid()
        plt.show()

    def _plot_profile(self, title: str, profile: Profile):

        # First subplot with the top view
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        axes[0].imshow(self.img)
        axes[0].set_xlabel("X [px]", fontweight="bold")
        axes[0].set_ylabel("Y [px]", fontweight="bold")
        axes[0].set_title("Top View of the crater", fontweight="bold")
        axes[0].plot(profile.x_pixel_bounds, profile.y_pixel_bounds, "ro-")
        axes[0].axis("image")
        ylim, xlim = self.img.shape
        axes[0].set_xlim(0, xlim)
        axes[0].set_ylim(0, ylim)
        if self.is_valid:
            axes[0].add_patch(self.ellipse.ellipse_patch())
            axes[0].scatter(*list(zip(*self.ellipse.landmarks)))

        # Second subplot with the profile view
        axes[1].plot(profile.s, profile.h)
        axes[1].set_title("Profile View of the crater", fontweight="bold")
        for x_bounds, z_bounds in profile.walls:
            axes[1].plot(x_bounds, z_bounds, color="blue", linestyle="--")  
        axes[1].set_ylabel("Depth [mm]", fontweight="bold")
        axes[1].set_xlabel("Distance [mm]", fontweight="bold")
        selected_indices = np.array(profile.key_indexes)
        axes[1].scatter(profile.s[selected_indices], profile.h[selected_indices])
        axes[1].grid()

        plt.show()