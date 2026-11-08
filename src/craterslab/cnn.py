from pathlib import Path

import cv2
import numpy as np

from craterslab.classification import SurfaceType
from craterslab.sensors import DepthMap

try:
    import keras
except ImportError as error:
    raise ImportError(
        "The CNN classifier needs the classification extra: "
        "pip install 'craterslab[classification]'"
    ) from error

# The surface type behind each output of the network, in order
CLASS_ORDER = (
    SurfaceType.UNKNOWN,
    SurfaceType.SIMPLE_CRATER,
    SurfaceType.COMPLEX_CRATER,
    SurfaceType.SAND_MOUND,
)
NUM_CLASSES = len(CLASS_ORDER)
IM_SIZE = 100
WEIGHTS_PATH = Path(__file__).parent.resolve() / "craterslab.weights.h5"


def get_raw_model() -> keras.Sequential:
    # Build a CNN
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IM_SIZE, IM_SIZE, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def get_untrained_model(lr: float = 0.001) -> keras.Sequential:
    model = get_raw_model()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_trained_model(model: keras.Sequential, path: Path | str = WEIGHTS_PATH) -> None:
    model.save_weights(path)


def normalize(img: np.ndarray, expand: bool = True) -> np.ndarray:
    resized = cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_CUBIC)
    max_val = max(abs(np.max(resized)), abs(np.min(resized)))
    result = resized / max_val
    if expand:
        return np.expand_dims(result, axis=0)
    return result


class CNNClassifier:
    """Classifies surfaces with a trained CNN. It expects auto-cropped depth maps."""

    def __init__(self, weights_path: Path | str = WEIGHTS_PATH) -> None:
        self.model = get_untrained_model()
        self.model.load_weights(weights_path)

    def classify(self, depth_map: DepthMap) -> SurfaceType:
        probabilities = self.model.predict(normalize(depth_map.map), verbose=0)
        return CLASS_ORDER[int(np.argmax(probabilities))]
