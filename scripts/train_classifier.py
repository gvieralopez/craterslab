import random
from pathlib import Path
from typing import Any

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from craterslab import DepthMap, SurfaceType
from craterslab.cnn import (
    CLASS_ORDER,
    NUM_CLASSES,
    CNNClassifier,
    get_untrained_model,
    normalize,
    save_trained_model,
)

DATA_PATH = Path("examples/data")

NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
CROP_BORDERS_BELOW = 0.25
AUTO_CROP_BELOW = 0.75
TRANSPOSE_ABOVE = 0.5

SIMPLE_INDICIES = [1, 2, 3, 4, 5, 6, 7, 8, 26, 27, 28, 32, 35]
COMPLEX_INDICIES = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Define the categories and their corresponding file names and ids
categories: dict[str, dict[str, Any]] = {
    "empty": {
        "files": [f"plane_{i}.npz" for i in range(1, 17)],
        "id": CLASS_ORDER.index(SurfaceType.UNKNOWN),
    },
    "simple_craters": {
        "files": [f"fluidized_{i}.npz" for i in SIMPLE_INDICIES],
        "id": CLASS_ORDER.index(SurfaceType.SIMPLE_CRATER),
    },
    "complex_craters": {
        "files": [f"fluidized_{i}.npz" for i in COMPLEX_INDICIES],
        "id": CLASS_ORDER.index(SurfaceType.COMPLEX_CRATER),
    },
    "sand_mounds": {
        "files": [f"compacted_{i}.npz" for i in range(1, 25)],
        "id": CLASS_ORDER.index(SurfaceType.SAND_MOUND),
    },
}


def regularize(dm: DepthMap) -> DepthMap:
    cropping_coin = random.random()
    if cropping_coin < CROP_BORDERS_BELOW:
        dm.crop_borders(ratio=cropping_coin)
    elif cropping_coin < AUTO_CROP_BELOW:
        dm.auto_crop()
    if random.random() > TRANSPOSE_ABOVE:
        dm.map = dm.map.T
    return dm


# Preprocess images and labels
image_list = []
label_list = []
for details in categories.values():
    for file in details["files"]:
        path = DATA_PATH / file
        dm = DepthMap.load(path)
        if details["id"]:
            dm = regularize(dm)
        img = normalize(dm.map, expand=False)
        image_list.append(img)
        label_list.append(details["id"])

# Convert images and labels to numpy arrays
images = np.expand_dims(np.array(image_list), axis=-1)
labels = to_categorical(np.array(label_list), num_classes=NUM_CLASSES)


# Split the data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2
)

# Train and save the model
model = get_untrained_model(lr=LEARNING_RATE)
model.fit(train_images, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
save_trained_model(model)

# Load the model from the file
model = CNNClassifier().model

# Test the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test set accuracy: {test_acc}")
