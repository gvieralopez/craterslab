# Craters Morphology Analysis Tool

A library to simplify the analysis of crater data from depth maps.

## Installation

Craterslab requires Python 3.10+ for functioning. Make sure you have a compliant version of [python](https://www.python.org/downloads/) installed in your system.

### Installing craterslab from the repository using poetry (Recommended)

Make sure you have [poetry](https://python-poetry.org/docs/) in your system and then run:

```
$ git clone https://github.com/gvieralopez/craterslab.git
$ cd craterslab
$ poetry install
$ poetry shell
```

These will create and activate an isolated virtual environment with craterslab installed on it. 

### Installing craterslab from pypi using pip

Craterslab is also available from pypi. You can install it by running:

```
$ pip install craterslab
```

We strongly encourage users using this method to create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) before installing the library.

## Usage

You can find some examples on how to use the library in this repository:

```
$ git clone https://github.com/gvieralopez/craterslab.git
$ cd craterslab/examples
```

Before executing any example, you will need to download data from actual craters using the provided scripts:

```
$ python download_data.py
```

Then, you can execute any given example as:

```
$ python example1.py
```

See [software documentation](https://craterslab.readthedocs.io/en/latest/) for more details.

