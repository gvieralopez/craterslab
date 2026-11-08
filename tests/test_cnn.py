import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np
import pytest

from craterslab.classification import SurfaceType
from craterslab.cnn import (
    CLASS_ORDER,
    IM_SIZE,
    NUM_CLASSES,
    CNNClassifier,
    get_raw_model,
    get_untrained_model,
    normalize,
    save_trained_model,
)
from craterslab.craters import Surface
from craterslab.depth_map import DepthMap


@pytest.fixture(scope="module")
def classifier() -> CNNClassifier:
    return CNNClassifier()


def test_every_surface_type_is_a_class() -> None:
    assert set(CLASS_ORDER) == set(SurfaceType)


def test_normalize_scales_and_adds_a_batch_axis() -> None:
    result = normalize(np.array([[-4.0, 2.0], [1.0, 0.0]]))
    assert result.shape == (1, IM_SIZE, IM_SIZE)
    assert np.abs(result).max() == pytest.approx(1.0)


def test_normalize_without_batch_axis() -> None:
    assert normalize(np.ones((5, 5)), expand=False).shape == (IM_SIZE, IM_SIZE)


def test_raw_model_output() -> None:
    assert get_raw_model().output_shape == (None, NUM_CLASSES)


@pytest.mark.parametrize(
    ("name", "surface_type"),
    [
        ("fluidized_1", SurfaceType.SIMPLE_CRATER),
        ("fluidized_10", SurfaceType.COMPLEX_CRATER),
        ("compacted_1", SurfaceType.SAND_MOUND),
        ("plane_10", SurfaceType.UNKNOWN),
    ],
)
def test_real_surfaces(
    real_map: Callable[[str], DepthMap],
    classifier: CNNClassifier,
    name: str,
    surface_type: SurfaceType,
) -> None:
    surface = Surface.from_depth_map(real_map(name))
    surface.classify(classifier)
    assert surface.type is surface_type


def test_save_and_reload_weights(tmp_path: Path) -> None:
    path = tmp_path / "model.weights.h5"
    model = get_untrained_model()
    save_trained_model(model, path)
    reloaded = CNNClassifier(path).model
    assert np.array_equal(model.get_weights()[0], reloaded.get_weights()[0])


def test_incompatible_weights_raise(tmp_path: Path) -> None:
    other = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(2)])
    path = tmp_path / "other.weights.h5"
    other.save_weights(path)
    with pytest.raises(ValueError):
        CNNClassifier(path)


def test_missing_weights_raise(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        CNNClassifier(tmp_path / "missing.weights.h5")


def test_import_without_the_extra_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keras", None)
    monkeypatch.delitem(sys.modules, "craterslab.cnn")
    with pytest.raises(ImportError, match=r"craterslab\[classification\]"):
        importlib.import_module("craterslab.cnn")
