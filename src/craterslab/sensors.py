from collections.abc import Sequence
from typing import Any

import numpy as np

from craterslab.depth_map import DepthMap, SensorResolution

KINECT_INSTALL = (
    "pip install https://github.com/sshh12/LibKinect2/releases/download/v0.1.0/"
    "libkinect2-0.1.0.tar.gz"
)
FRAME_ATTEMPTS_PER_FRAME = 10


def capture_kinect(resolution: SensorResolution, average_on: int) -> DepthMap:
    try:
        from libkinect2 import Kinect2  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(f"The Kinect2 driver is not installed: {KINECT_INSTALL}") from error

    kinect = Kinect2(use_sensors=["depth"])
    kinect.connect()
    kinect.wait_for_worker()
    frames = [np.squeeze(next(kinect.iter_frames())[1]) for _ in range(average_on)]
    kinect.disconnect()
    return DepthMap(average_valid_frames(frames), resolution)


def capture_femto_bolt(resolution: SensorResolution, average_on: int, timeout_ms: int) -> DepthMap:
    try:
        import pyorbbecsdk2 as sdk  # noqa: PLC0415
    except ImportError as error:
        raise ImportError("The Femto Bolt needs its SDK: pip install pyorbbecsdk2") from error

    pipeline = sdk.Pipeline()
    config = sdk.Config()
    profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
    config.enable_stream(profiles.get_default_video_stream_profile())

    frames: list[np.ndarray] = []
    pipeline.start(config)
    try:
        for _ in range(average_on * FRAME_ATTEMPTS_PER_FRAME):
            frame_set = pipeline.wait_for_frames(timeout_ms)
            depth_frame = None if frame_set is None else frame_set.get_depth_frame()
            if depth_frame is not None:
                frames.append(_femto_depth_array(depth_frame))
            if len(frames) == average_on:
                break
    finally:
        pipeline.stop()
    if len(frames) < average_on:
        raise TimeoutError(f"The Femto Bolt gave {len(frames)} of {average_on} depth frames")
    return DepthMap(average_valid_frames(frames), resolution)


def average_valid_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Average frames ignoring zeros, the value depth sensors give to invalid pixels."""
    if not frames:
        raise ValueError("There are no depth frames to average")
    stack = np.stack(frames).astype(np.float64)
    valid = stack > 0
    count = valid.sum(axis=0)
    result = np.zeros(stack.shape[1:], dtype=np.float64)
    np.divide(np.where(valid, stack, 0.0).sum(axis=0), count, out=result, where=count > 0)
    return result


def _femto_depth_array(depth_frame: Any) -> np.ndarray:
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    shaped = raw.reshape((depth_frame.get_height(), depth_frame.get_width()))
    scale = (
        depth_frame.get_depth_scale()
        if hasattr(depth_frame, "get_depth_scale")
        else depth_frame.get_value_scale()
    )
    depth: np.ndarray = shaped.astype(np.float64) * float(scale)
    return depth
