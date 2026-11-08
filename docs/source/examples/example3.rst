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

   from craterslab import DepthMap
   depth_map = DepthMap.load("fluidized_1.npz")
   depth_map.auto_crop()


Creating a Surface
------------------

Surface objects are created from a depth map. If you already know what the 
surface contains, you can say so:

.. code-block:: python

   from craterslab import SurfaceType
   from craterslab import Surface
   s = Surface.from_depth_map(depth_map, SurfaceType.SIMPLE_CRATER)

Without a type, the surface is Unknown. An Unknown surface has no elliptical 
model and no observables until it gets a type:

.. code-block:: python

   s = Surface.from_depth_map(depth_map)
   s.set_type(SurfaceType.SIMPLE_CRATER)


Classification
--------------

Craterslab can also find the morphology for you with its CNN classifier, which 
needs the optional extra (see the installation page) and expects an auto-cropped
depth map:

.. code-block:: python

   from craterslab.cnn import CNNClassifier
   s = Surface.from_depth_map(depth_map)
   s.classify(CNNClassifier())
   print(s.type)

Any object with a ``classify(depth_map)`` method that returns a ``SurfaceType`` 
can be used instead.

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
   Maximum height (H_m): 3.00 mm
   Mean Height over the rim (mean_h_rim): 1.61 mm
   Concavity Volume (V_in): 65334.24 mm³
   Excavated Volume (V_ex): 27954.53 mm³
   """

Saving and loading
------------------

A surface can be saved along with its type and its elliptical models. Loading it 
needs neither the classifier nor a new fit, so the values stay the same on any 
machine:

.. code-block:: python

   s.save("my_crater.npz")
   s = Surface.load("my_crater.npz")

The file can also be read as a plain depth map with ``DepthMap.load``. In the 
other direction, ``Surface.load`` gives an Unknown surface for a file saved from
a depth map.

Exporting to CSV
----------------

The observables of many surfaces can be written to one CSV file, with one row 
per surface. Observables that do not apply to a surface are left empty. A 
profile can be exported too:

.. code-block:: python

   from craterslab import write_observables_csv, write_profile_csv
   write_observables_csv("observables.csv", {"crater_1": s})
   write_profile_csv("profile.csv", s.max_profile)
