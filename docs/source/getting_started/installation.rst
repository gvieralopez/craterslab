Installation and compatibility
==============================

Craterslab requires Python 3.11+ for functioning. Make sure you have a compliant 
version of `python <https://www.python.org/downloads/>`_ installed in your system.

Installing craterslab from pypi using pip (Recommended) 
-------------------------------------------------------

>>> pip install craterslab

We strongly encourage users using this method to create a 
`virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_ 
before installing the library.


Automatic classification
------------------------

The core install has no machine learning dependencies. To classify surfaces
with the bundled CNN, install the optional extra:

>>> pip install 'craterslab[classification]'

It works on Python 3.11 to 3.13, because TensorFlow has no wheels for newer
versions yet.

Installing craterslab from the repository using uv
--------------------------------------------------
Make sure you have `uv <https://docs.astral.sh/uv/getting-started/installation/>`_
in your system and then run:

>>> git clone https://github.com/gvieralopez/craterslab.git
>>> cd craterslab
>>> uv sync

These will create an isolated virtual environment with craterslab installed on
it. Run commands inside it with ``uv run``.
