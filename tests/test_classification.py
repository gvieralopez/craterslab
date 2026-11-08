import logging
from pathlib import Path

import numpy as np
import pytest

from craterslab import classification
from craterslab.classification import (
    IM_SIZE,
    NUM_CLASSES,
    SurfaceType,
    get_raw_model,
    get_trained_model,
    get_untrained_model,
    normalize,
    save_trained_model,
)


def test_surface_type_names() -> None:
    assert str(SurfaceType.COMPLEX_CRATER) == "Complex crater"
    assert len(SurfaceType) == NUM_CLASSES


def test_normalize_scales_and_adds_a_batch_axis() -> None:
    result = normalize(np.array([[-4.0, 2.0], [1.0, 0.0]]))
    assert result.shape == (1, IM_SIZE, IM_SIZE)
    assert np.abs(result).max() == pytest.approx(1.0)


def test_normalize_without_batch_axis() -> None:
    assert normalize(np.ones((5, 5)), expand=False).shape == (IM_SIZE, IM_SIZE)


def test_raw_model_output() -> None:
    assert get_raw_model().output_shape == (None, NUM_CLASSES)


def test_bundled_weights_load() -> None:
    model = get_trained_model()
    assert model is not None
    assert model.predict(np.zeros((1, IM_SIZE, IM_SIZE, 1)), verbose=0).shape == (1, NUM_CLASSES)


def test_save_and_reload_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classification, "CACHE_PATH", str(tmp_path / "model.weights.h5"))
    model = get_untrained_model()
    save_trained_model(model)
    reloaded = get_trained_model()
    assert reloaded is not None
    assert np.array_equal(model.get_weights()[0], reloaded.get_weights()[0])


def test_incompatible_weights_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    other = tmp_path / "other.weights.h5"
    get_untrained_model().save_weights(other)
    monkeypatch.setattr(classification, "CACHE_PATH", str(other))
    monkeypatch.setattr(classification, "NUM_CLASSES", NUM_CLASSES + 1)
    with caplog.at_level(logging.ERROR):
        assert get_trained_model() is None
    assert "could not be loaded" in caplog.text
