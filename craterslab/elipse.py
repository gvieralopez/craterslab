import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import Bounds, minimize


def fit_elipse(
    x: np.ndarray,
    y: np.ndarray,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
) -> tuple:

    # Define the objective function for fitting the ellipse
    def objective(params, x, y):
        a, b, cx, cy, theta = params
        ct = np.cos(theta)
        st = np.sin(theta)
        x_hat = ct * (x - cx) + st * (y - cy)
        y_hat = -st * (x - cx) + ct * (y - cy)
        ellipse = ((x_hat / a) ** 2 + (y_hat / b) ** 2) - 1
        return np.sum(ellipse**2)

    # Compute the mean and standard deviation of the data
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)

    # Define the initial guess for the ellipse parameters based on the mean and standard deviation of the data
    params0 = [x_std, y_std, x_mean, y_mean, 0]

    # Define the bounds for the ellipse
    lower_bound = [xrange[0], yrange[0], xrange[0], yrange[0], 0]
    upper_bound = [xrange[1], yrange[1], xrange[1], yrange[1], 2 * np.pi]
    bounds = Bounds(np.array(lower_bound), np.array(upper_bound))

    # Fit the ellipse to the points using the objective function and the initial guess
    result = minimize(objective, params0, args=(x, y), bounds=bounds)

    # Extract the fitted ellipse parameters from the optimization result, and create the `Ellipse` patch using these parameters
    return result.x


if __name__ == "__main__":
    points = [
        (118.65465465465465, 76.0),
        (33.82582582582583, 76.00000000000001),
        (108.87687687687688, 107.18318318318319),
        (37.525525525525495, 55.2912912912913),
        (80.43943943943944, 119.8958958958959),
        (53.2122122122122, 37.12512512512512),
        (53.76276276276276, 113.2012012012012),
        (80.08908908908909, 33.16916916916918),
        (39.37537537537538, 95.36336336336336),
        (105.30930930930931, 47.41141141141142),
    ]
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    a, b, cx, cy, theta = fit_elipse(x, y, (0, 500), (0, 500))

    ellipse_patch = Ellipse((cx, cy), 2 * a, 2 * b, theta * 180 / np.pi, fill=False)

    # Plot the points and the fitted ellipse
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.add_patch(ellipse_patch)
    ax.set_aspect("equal")
    plt.show()
