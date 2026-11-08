import importlib
import logging

import pytest

import craterslab


@pytest.mark.parametrize("name", craterslab.__all__)
def test_the_public_names_exist(name: str) -> None:
    assert hasattr(craterslab, name)


def test_the_ml_classifier_stays_optional() -> None:
    assert "CNNClassifier" not in craterslab.__all__


def test_importing_does_not_configure_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: object) -> None:
        raise AssertionError("the library configured the logging of its host")

    monkeypatch.setattr(logging, "basicConfig", fail)
    importlib.reload(craterslab)


def test_the_library_logger_is_silent_by_default() -> None:
    handlers = logging.getLogger("craterslab").handlers
    assert any(isinstance(handler, logging.NullHandler) for handler in handlers)
