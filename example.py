from lidar import load_from_file
from craters import Crater

# Define data sources
after_impact_data = load_from_file('craterexp1.mat')
before_impact_data = load_from_file('planoexp1.mat')

# Define physical constants
image_resolution = 2.8025  # Distance in mm between adjacent pixels 
image_depth = 1.0  # Amount of mm on every increment on image color

# Create the Crater object
c = Crater(before_impact_data, after_impact_data, image_resolution, image_depth)

# Plot the crater in 3D
preview_scale = (1, 1, 10)
c.plot_3D(title='Crater view in 3D', preview_scale=preview_scale)

# Plot a transversal cut of the crater
c.set_profile(start_point=(0, 0), end_point=(100, 100))
c.plot_profile('Crater profile')
