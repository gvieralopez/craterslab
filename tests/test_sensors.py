import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from craterslab.depth_map import SensorResolution
from craterslab.sensors import average_valid_frames, capture_femto_bolt, capture_kinect


def test_average_ignores_invalid_pixels() -> None:
    frames = [np.array([[2.0, 0.0]]), np.array([[4.0, 6.0]])]
    assert average_valid_frames(frames).tolist() == [[3.0, 6.0]]


def test_average_keeps_zero_where_a_pixel_is_never_valid() -> None:
    assert average_valid_frames([np.zeros((1, 2)), np.zeros((1, 2))]).tolist() == [[0.0, 0.0]]


def test_average_accepts_integer_frames() -> None:
    frames = [np.array([[1, 2]], dtype=np.uint16), np.array([[2, 2]], dtype=np.uint16)]
    assert average_valid_frames(frames).tolist() == [[1.5, 2.0]]


def test_average_needs_frames() -> None:
    with pytest.raises(ValueError, match="no depth frames"):
        average_valid_frames([])


class _Kinect:
    disconnected = False

    def __init__(self, use_sensors: list[str]) -> None:
        self.frames = iter(
            [np.array([[[2.0], [0.0]], [[1.0], [1.0]]]), np.array([[[4.0], [6.0]], [[3.0], [1.0]]])]
        )

    def connect(self) -> None: ...

    def wait_for_worker(self) -> None: ...

    def iter_frames(self) -> Iterator[tuple[None, np.ndarray]]:
        yield None, next(self.frames)

    def disconnect(self) -> None:
        _Kinect.disconnected = True


def test_kinect_averages_valid_pixels(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    monkeypatch.setitem(sys.modules, "libkinect2", SimpleNamespace(Kinect2=_Kinect))
    depth_map = capture_kinect(resolution, average_on=2)
    assert depth_map.map.tolist() == [[3.0, 6.0], [2.0, 1.0]]
    assert depth_map.sensor is resolution
    assert _Kinect.disconnected


def test_kinect_without_its_driver_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    monkeypatch.setitem(sys.modules, "libkinect2", None)
    with pytest.raises(ImportError, match="LibKinect2"):
        capture_kinect(resolution, average_on=1)


class _FemtoFrame:
    def __init__(self, values: list[int], **scale: Any) -> None:
        self.values = np.array(values, dtype=np.uint16)
        for name, value in scale.items():
            setattr(self, name, lambda value=value: value)

    def get_data(self) -> bytes:
        return self.values.tobytes()

    def get_height(self) -> int:
        return 1

    def get_width(self) -> int:
        return len(self.values)


class _FemtoPipeline:
    def __init__(self, frames: list[Any]) -> None:
        self.frames = iter(frames)
        self.started = self.stopped = False
        self.timeouts: list[int] = []

    def get_stream_profile_list(self, sensor: str) -> SimpleNamespace:
        return SimpleNamespace(get_default_video_stream_profile=lambda: f"profile of {sensor}")

    def start(self, config: SimpleNamespace) -> None:
        self.started = config.stream == "profile of depth"

    def wait_for_frames(self, timeout_ms: int) -> Any:
        self.timeouts.append(timeout_ms)
        return next(self.frames)

    def stop(self) -> None:
        self.stopped = True


def _install_femto(monkeypatch: pytest.MonkeyPatch, frames: list[Any]) -> _FemtoPipeline:
    pipeline = _FemtoPipeline(frames)
    config = SimpleNamespace(stream=None)
    config.enable_stream = lambda profile: setattr(config, "stream", profile)
    sdk = SimpleNamespace(
        Pipeline=lambda: pipeline,
        Config=lambda: config,
        OBSensorType=SimpleNamespace(DEPTH_SENSOR="depth"),
    )
    monkeypatch.setitem(sys.modules, "pyorbbecsdk2", sdk)
    return pipeline


def _frame_set(depth_frame: Any) -> SimpleNamespace:
    return SimpleNamespace(get_depth_frame=lambda: depth_frame)


def test_femto_scales_and_averages_valid_pixels(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    frames = [
        _frame_set(_FemtoFrame([2, 0], get_depth_scale=0.5)),
        _frame_set(_FemtoFrame([4, 6], get_depth_scale=0.5)),
    ]
    pipeline = _install_femto(monkeypatch, frames)
    depth_map = capture_femto_bolt(resolution, average_on=2, timeout_ms=100)
    assert depth_map.map.tolist() == [[1.5, 3.0]]
    assert pipeline.started
    assert pipeline.stopped
    assert pipeline.timeouts == [100, 100]


def test_femto_supports_the_older_scale_name(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    _install_femto(monkeypatch, [_frame_set(_FemtoFrame([4], get_value_scale=0.25))])
    assert capture_femto_bolt(resolution, 1, 100).map.tolist() == [[1.0]]


def test_femto_skips_missing_frames(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    frames = [None, _frame_set(None), _frame_set(_FemtoFrame([8], get_depth_scale=1.0))]
    _install_femto(monkeypatch, frames)
    assert capture_femto_bolt(resolution, 1, 100).map.tolist() == [[8.0]]


def test_femto_gives_up_when_no_frames_arrive(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    pipeline = _install_femto(monkeypatch, [None] * 10)
    with pytest.raises(TimeoutError, match="0 of 1"):
        capture_femto_bolt(resolution, 1, 100)
    assert pipeline.stopped


def test_femto_stops_the_pipeline_after_an_error(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    pipeline = _install_femto(monkeypatch, [])
    with pytest.raises(StopIteration):
        capture_femto_bolt(resolution, 1, 100)
    assert pipeline.stopped


def test_femto_without_its_sdk_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, resolution: SensorResolution
) -> None:
    monkeypatch.setitem(sys.modules, "pyorbbecsdk2", None)
    with pytest.raises(ImportError, match="pip install pyorbbecsdk2"):
        capture_femto_bolt(resolution, 1, 100)
