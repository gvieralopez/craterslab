Analysis of surfaces
====================

Beyond processing and visualizing generic depth maps, craterslab is able to 
recognize and characterize specific morphologies from the data contained in 
the depth maps. To do so, the library introduces a more abstract concept named 
Surface. It is a sort of depth map's analyzer. 

For all the plots shown we will be using 
`this depth map <https://github.com/gvieralopez/craters-data/blob/main/data/fluidized_1.npz>`_, 
which can be directly loaded with craterslab as:

.. code-block:: python

   from craterslab.sensors import DepthMap
   depth_map = DepthMap.load("fluidized_1.npz")
   depth_map.auto_crop()


Creating a Surface
------------------

Surface objects can be created by simply passing a depth map:

.. code-block:: python

   from craterslab.craters import Surface
   s = Surface(depth_map)


Classification
--------------

Now, we can see if there is any known morphology in the depth map by:

.. code-block:: python

   print(s.type)

This should output one of the following:

- Simple Crater 
- Complex Crater
- Sand Mound
- Unknown


Quantifying its observables
---------------------------

For all the Surface objects with types different than 'Unknown' there are 
several observables that are computed in order to better characterize its 
morphology. In this example, we are using a depth map containing a simple 
crater. We can check the value of its observables by:


.. code-block:: python

   print(s)

   """
   Found: Simple crater

   Apparent Depth (d_max): -15.91 mm
   Eccentricity (epsilon): 0.38 
   Diameter (D): 99.19 mm
   Maximum heigh (H_cp): 3.00 mm
   Mean Heigh over the rim (mean_h_rim): 1.61 mm
   Concavity Volume (V_in): 65334.24 mm³
   Excavated Volume (V_ex): 27954.53 mm³
   """

