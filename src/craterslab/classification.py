import logging
from enum import Enum
from pathlib import Path

import cv2
import keras
import numpy as np
import keras

class SurfaceType(Enum):
    UNKNOWN = 0
    SIMPLE_CRATER = 1
    COMPLEX_CRATER = 2
    SAND_MOUND = 3

    def __str__(self):
        return self.name.replace("_", " ").capitalize()


NUM_CLASSES = len(SurfaceType)
IM_SIZE = 100
CACHE_PATH = str(Path(__file__).parent.resolve() / "craterslab.weights.h5")


def get_raw_model() -> keras.Sequential:
    # Build a CNN
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IM_SIZE, IM_SIZE, 1)
        )
    )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    return model

def get_untrained_model(lr=0.001) -> keras.Sequential:
    model = get_raw_model()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_trained_model() -> keras.Sequential:
    try:
        model = get_untrained_model()
        model.load_weights(CACHE_PATH)
        return model
    except ValueError:
        logging.error("Pre-trained classification model could not be loaded")


def save_trained_model(model):
    model.save_weights(CACHE_PATH)


def normalize(img: np.ndarray, expand=True) -> np.ndarray:
    resized = cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_CUBIC)
    max_val = max(abs(np.max(resized)), abs(np.min(resized)))
    result = resized / max_val
    if expand:
        return np.expand_dims(result, axis=0)
    return result
