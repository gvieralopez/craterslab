<div align="center">

<img src="assets/logo.svg" alt="" width="96">

# craterslab

**A library to simplify the analysis of crater data from depth maps.**

[![QA](https://github.com/gvieralopez/craterslab/actions/workflows/qa.yml/badge.svg)](https://github.com/gvieralopez/craterslab/actions/workflows/qa.yml)
[![Release](https://img.shields.io/github/v/release/gvieralopez/craterslab?label=release)](https://github.com/gvieralopez/craterslab/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://readthedocs.org/projects/craterslab/badge/?version=latest)](https://craterslab.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/craterslab)](https://pypi.org/project/craterslab/)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

<!-- Badges to move into the rows above once they apply:
[![Docker image](https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/gvieralopez/craterslab/pkgs/container/craterslab)
-->

[Installation](#installation) · [Usage](#usage) · [Troubleshooting](#troubleshooting) · [Development](#development)

</div>

---

## Installation

Craterslab requires Python 3.11+ for functioning. Make sure you have a compliant version of [python](https://www.python.org/downloads/) installed in your system.

### Installing craterslab from pypi using pip (Recommended)


Craterslab is also available from pypi. You can install it by running:

```
$ pip install craterslab
```

We strongly encourage users using this method to create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) before installing the library.

### Automatic classification

The core install has no machine learning dependencies. To classify surfaces with the bundled CNN, install the optional extra:

```
$ pip install 'craterslab[classification]'
```

It works on Python 3.11 to 3.13, because TensorFlow has no wheels for newer versions yet.

### Installing craterslab from the repository using uv
Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) in your system and then run:

```
$ git clone https://github.com/gvieralopez/craterslab.git
$ cd craterslab
$ uv sync
```

These will create an isolated virtual environment with craterslab installed on it. Run commands inside it with `uv run`.

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

## Troubleshooting

### Error running the examples

If you get a warning as such

```
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

You may need to install tk on your operative system.

## Development

Setup, QA, docs and releases: [DEVELOPMENT.md](DEVELOPMENT.md).

## Contributing

Issues and pull requests are welcome; `make qa` must pass.

## License

MIT: see [LICENSE](LICENSE).

---

<sub>🚀 Project starter provided by [Cookie Pyrate v1.1.2](https://github.com/gvieralopez/cookie-pyrate/tree/v1.1.2)</sub>
