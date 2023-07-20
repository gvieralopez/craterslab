from enum import Enum
from pathlib import Path

import cv2
import keras
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


class SurfaceType(Enum):
    UNKNOWN = 0
    SIMPLE_CRATER = 1
    COMPLEX_CRATER = 2
    SAND_MOUND = 3

    def __str__(self):
        return self.name.replace("_", " ").capitalize()


NUM_CLASSES = len(SurfaceType)
IM_SIZE = 100
CACHE_PATH = Path(__file__).parent.resolve() / "surface_classifier.keras"


def get_untrained_model(lr=0.001) -> Sequential:
    # Build a CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IM_SIZE, IM_SIZE, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    # Compile the model
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_trained_model() -> Sequential:
    return keras.models.load_model(CACHE_PATH)


def save_trained_model(model):
    model.save(CACHE_PATH)


def normalize(img: np.ndarray, expand=True) -> np.ndarray:
    resized = cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_CUBIC)
    max_val = max(abs(np.max(resized)), abs(np.min(resized)))
    result = resized / max_val
    if expand:
        return np.expand_dims(result, axis=0)
    return result
     
