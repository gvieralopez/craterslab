from pycraters.lidar import load_from_file
from pycraters.craters import Crater

# Define data sources
after_impact_data = load_from_file('crater_aFig4.mat')
before_impact_data = load_from_file('plano_aFig4.mat')

# Define physical constants
image_resolution = 2.8025  # Distance in mm between adjacent pixels 
image_depth = 1.0  # Amount of mm on every increment on image color

# Create the Crater object
c = Crater(before_impact_data, after_impact_data, image_resolution, image_depth)

# Plot the crater in 3D
c.plot_3D(title='Crater view in 3D', preview_scale=(1, 1, 4))

# Plot a transversal cut of the crater
#c.plot_profile('Crater profile')

# Inspect observables
print(c)
