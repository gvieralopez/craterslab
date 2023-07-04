from pycraters.lidar import load_from_file
from pycraters.craters import Crater


# Define physical constants
image_resolution = 2.8025  # Distance in mm between adjacent pixels 
image_depth = 1.0  # Amount of mm on every increment on image color

for i in range(1, 30):

    print(i)

    # Define data sources
    after_impact_data = load_from_file(f'craterexp{i}.mat')
    before_impact_data = load_from_file(f'planoexp{i}.mat')

    # Create the Crater object
    c = Crater(before_impact_data, after_impact_data, image_resolution, image_depth)

    # Plot the crater in 3D
    c.plot_3D(title='Crater view in 3D', preview_scale=(1, 1, 4))

    # Plot a transversal cut of the crater
    c.plot_profile('Crater profile')

    # Inspect observables
    print(c)
