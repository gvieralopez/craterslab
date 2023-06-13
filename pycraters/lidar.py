import pathlib
import numpy as np
import scipy.io


def _load_file(filename: str, data_folder: str):
    # Path to the desired file
    file_path = pathlib.Path(data_folder, filename)

    # Load the .mat files
    mat_contents = scipy.io.loadmat(file_path)

    # Access the data in the .mat file
    return mat_contents[filename[:-4]]


def load_from_file(
    filename: str,
    data_folder: str = "data",
    max_samples: int = -1,
    average: bool = True,
):
    # Retrieve the content of the .mat file
    content = _load_file(filename, data_folder)

    # Reduce the amount of raw data used
    if max_samples != -1:
        content = content[:, :, :, :max_samples]

    # Average the last dimension to obtain an image-like ndarray
    if average:
        content = np.mean(content, axis=-1).squeeze()

    return content
