import logging
import random

import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from craterslab.classification import (
    NUM_CLASSES,
    get_trained_model,
    get_untrained_model,
    normalize,
    save_trained_model,
)
from craterslab.sensors import DepthMap

NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.0001

# Define the categories and their corresponding data folders and file indices
categories = {
    "empty": {
        "folder": "data/Fluized_sand",
        "indices": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "load": "single",
        "id": 0,
    },
    "simple_craters": {
        "folder": "data/Fluized_sand",
        "indices": [1, 2, 3, 4, 5, 6, 7, 8, 26, 27, 28, 32, 35],
        "load": "differential",
        "id": 1,
    },
    "complex_craters": {
        "folder": "data/Fluized_sand",
        "indices": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        + [29, 30, 31, 33, 34, 36, 37],
        "load": "differential",
        "id": 2,
    },
    "sand_mounds": {
        "folder": "data/Compacted_sand",
        "indices": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "load": "differential",
        "id": 3,
    },
}


def load(folder, index, mode):
    if mode == "single":
        depth_map = DepthMap.from_mat_file(f"planoexp{index}.mat", data_folder=folder)
    elif mode == "differential":
        d0 = DepthMap.from_mat_file(f"planoexp{index}.mat", data_folder=folder)
        df = DepthMap.from_mat_file(f"craterexp{index}.mat", data_folder=folder)
        depth_map = d0 - df
    else:
        logging.error(f"Unknown load mode: {mode}")

    return depth_map


def regularize(dm: DepthMap) -> DepthMap:
    cropping_coin = random.random()
    if cropping_coin < 0.25:
        dm.crop_borders(ratio=cropping_coin)
    elif cropping_coin < 0.75:
        dm.auto_crop()
    if random.random() > 0.5:
        dm.map = dm.map.T
    return dm


# Preprocess images and labels
images = []
labels = []
for category, details in categories.items():
    for index in details["indices"]:
        dm = load(details["folder"], index, details["load"])
        if details["id"]:
            dm = regularize(dm)
        img = normalize(dm.map, expand=False)
        images.append(img)
        labels.append(details["id"])

# Convert images and labels to numpy arrays
images = np.expand_dims(np.array(images), axis=-1)
labels = to_categorical(np.array(labels), num_classes=NUM_CLASSES)


# Split the data into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2
)

# Train and save the model
model = get_untrained_model(lr=LEARNING_RATE)
model.fit(train_images, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
save_trained_model(model)

# Load the model from the file
model = get_trained_model()

# Test the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test set accuracy: {test_acc}")
