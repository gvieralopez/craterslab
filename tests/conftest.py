from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import pyplot as plt

from craterslab.sensors import DepthMap, SensorResolution

mpl.use("Agg")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.fixture
def resolution() -> SensorResolution:
    return SensorResolution(1.0, 1.0, 1.0)


@pytest.fixture
def crater(resolution: SensorResolution) -> DepthMap:
    """A round synthetic crater: bowl of radius 30, rim of height 2, depth 10 below the rim."""
    size, radius, depth, rim = 101, 30.0, 10.0, 2.0
    y, x = np.mgrid[:size, :size]
    r = np.hypot(x - size // 2, y - size // 2)
    bowl = rim - depth * (1 - (r / radius) ** 2)
    outside = rim * np.exp(-(((r - radius) / 8) ** 2))
    return DepthMap(np.where(r < radius, bowl, outside), resolution)


@pytest.fixture
def real_map() -> Callable[[str], DepthMap]:
    def load(name: str) -> DepthMap:
        dm = DepthMap.load(FIXTURES / f"{name}.npz")
        dm.auto_crop()
        return dm

    return load
