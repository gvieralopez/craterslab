Operations with depth maps
==========================

A depth map is a 2D image that contains information about the distance between 
the camera and the objects in a scene. It represents the spatial layout of a 
scene in terms of its depth, providing a way to create a 3D representation of 
the scene. Depth maps can be generated through various techniques, such as 
stereo imaging, structured light scanning, or time-of-flight measurements.
In craterslab, depth maps are the core data structure used for the analysis of
the morphology of surfaces.

This examples illustrates how to retrieve depth maps from different sources. 
Then, it will guide you through storing and loading already created depth maps.
Finally, optional pre-processing operations for cropping depth maps are 
illustrated.

.. _kinect 1:

Capture depth maps from the Kinect sensor
-----------------------------------------

Fetching depth maps from the Kinect v2 sensor is directly supported with 
craterslab. As in any application that required to estimate real world scales
from images, the first step is to estimate the spacial resolution of the 
kinect, which directly depends on the distance you are placing it from the 
scene. So, you will need to estimate how many real world units (e.g., 
millimeters) are contained on each pixel's heigh and width. In the following
example, this value was estimated to 2.8025 mm/pixel. In the case of the 
z axis resolution, the sensor will always deliver a resolution of 1 unit/mm.

With all these values, we can create a SensorResolution object, which will 
define the intrinsic values from the sensor and it will allow the library to 
map the depth maps into real world units.

Finally, we create the DepthMap object using the from_kinect_sensor method, 
passing it the aforementioned resolution and, optionally, the number of 
independent shots to take if we want a depth map averaged over time.

The resulting code would be:

.. code-block:: python

   from craterslab.sensors import DepthMap, SensorResolution
   resolution = SensorResolution(2.8025, 2.8025, 1.0, 'mm')
   depth_map = DepthMap.from_kinect_sensor(resolution, average_on=500)

.. _cloud points 1:

Create depth maps from cloud points in .xyz format
--------------------------------------------------

When data is in the form of a cloud point, we need to convert it into a 
depth map first. To do so, we have to map each points into a matrix whose values
represent the z value from the points and the indices i,j from the matrix will
be proportional to their x and y coordinates respectively. So, the x coordinate
form the points will be determined by i * M while the y coordinate will equal
j * N, where M and N are the spacial resolution of the depth map on each axis
respectively.

Similar to the previous example, we can define the desired resolution of the 
depth map by:


.. code-block:: python

   data_resolution = SensorResolution(235.65, 235.65, 1.0, "m")

where the last parameter establishes the scale in which all computations will
be made and should be chosen accordingly with the original scale from the point
cloud. Then, the first two parameters define M and N respectively, and can be 
thought of as the number of units in real scale (meters in this case) between 
two consecutive pixels from the depth map. The third parameter accounts from the 
scale in the z axis (how many sensor units equal one real scale unit). 

Then, we can use this resolution to create a DepthMap as:

.. code-block:: python

   depth_map = DepthMap.from_xyz_file("king.xyz", resolution=data_resolution)

In the previous example, we are using 
`this xyz file <https://github.com/gvieralopez/craters-data/blob/main/data/king.xyz>`_.
with a three dimensional cloud point from `the lunar crater King <https://en.wikipedia.org/wiki/King_(crater)>`_.

.. _load save 1:

Saving and Loading depth maps in craterslab format
--------------------------------------------------

Regardless the method used to acquire a depth map, it is always possible to 
store it in a file. Craterslab uses numpy's .npz format to store the matrix
containing the depth map along with the sensor resolution associated.

.. code-block:: python

   depth_map.save('my_depth_map.npz')

Then, any depth map saved with craters lab can be recovered by:

.. code-block:: python

   depth_map = DepthMap.load('my_depth_map.npz')

In `this repository  <https://github.com/gvieralopez/craters-data>`_ there are 
several depth maps that can be directly imported using craterslab. Those depth 
maps were captured from granular surfaces where craters are present.

.. _crop 1:

Cropping depth maps
-------------------

Very frequently, captured depth maps contain much information outside the region 
of interest for an specific application. In those cases, craterslab provides
different methods to crop the depth maps to the desired region.

First, you can manually specify the bounding box (in pixel coordinates) you want 
to preserve by:

.. code-block:: python

   bounding_box = (0, 0, 100, 100) 
   depth_map.crop(bounding_box)

If your depth_map is perfectly centered, but you want to crop the borders, you 
can use the crop_borders method, specifying a ratio from 0 to 1, where 0 means 
no cropping and 1 means crop the entire image.

.. code-block:: python

   depth_map.crop_borders(0.5)

Finally, you can let craterslab crop the depth_map for you. It will identify 
regions of significant variability and keep only them in the resulting depth 
map:

.. code-block:: python

   depth_map.auto_crop()


Subtracting depth maps
----------------------

In some scenarios it is convenient to store the depth map resulting from the 
subtraction of two other depth maps taken from the same surface at different
time instants. For instance, when studying impact craters made under laboratory
conditions, we can take a depth map from the surface before the impact and 
another from after the impact. That can be achieved with craters lab by:

.. code-block:: python

   import time
   d0 = DepthMap.from_kinect_sensor(resolution, average_on=500)
   time.sleep(60 * 5) # Five minutes break to produce the crater
   df = DepthMap.from_kinect_sensor(resolution, average_on=500)
   depth_map = d0 - df