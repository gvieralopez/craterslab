import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import minimize

def fit_elipse(x: np.ndarray, y: np.ndarray) -> tuple:

    # Define the objective function for fitting the ellipse
    def objective(params, x, y):
        a, b, cx, cy, theta = params
        ct = np.cos(theta)
        st = np.sin(theta)
        x_hat = ct * (x - cx) + st * (y - cy)
        y_hat = -st * (x - cx) + ct * (y - cy)
        ellipse = ((x_hat / a)**2 + (y_hat / b)**2) - 1
        return np.sum(ellipse**2)

    # Define the initial guess for the ellipse parameters
    params0 = [1, 1, 0, 0, 0]

    # Fit the ellipse to the points using the objective function and the initial guess
    result = minimize(objective, params0, args=(x, y))

    # Extract the fitted ellipse parameters from the optimization result, and create the `Ellipse` patch using these parameters
    return result.x

if __name__ == "__main__":
    points = [(93.58558558558559, 50.0), (8.806806806806819, 50.000000000000014), (88.49249249249249, 76.11711711711712), (15.067067067067057, 22.08708708708707), (66.37337337337337, 91.7917917917918), (39.114114114114116, 6.60660660660659), (39.850850850850854, 91.09109109109109), (65.63663663663664, 10.5105105105105), (17.825825825825827, 75.88288288288288), (84.24824824824827, 27.006006006005997)]
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    a, b, cx, cy, theta = fit_elipse(x, y)

    ellipse_patch = Ellipse((cx, cy), 2*a, 2*b, theta * 180/np.pi, fill=False)

    # Plot the points and the fitted ellipse
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.add_patch(ellipse_patch)
    ax.set_aspect('equal')
    plt.show()