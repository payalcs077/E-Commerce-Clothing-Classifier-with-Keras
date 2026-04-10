from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from config import FASHION_MNIST_CACHE_FILE, FASHION_MNIST_DATASET_SLUG


def download_dataset_dir() -> Path:
    return Path(kagglehub.dataset_download(FASHION_MNIST_DATASET_SLUG))


def load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(csv_path)
    labels = dataframe["label"].to_numpy(dtype=np.int64)
    images = dataframe.drop(columns=["label"]).to_numpy(dtype=np.uint8)
    images = images.reshape(-1, 28, 28)
    return images, labels


def build_cache(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels = load_csv_dataset(dataset_dir / "fashion-mnist_train.csv")
    test_images, test_labels = load_csv_dataset(dataset_dir / "fashion-mnist_test.csv")

    FASHION_MNIST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FASHION_MNIST_CACHE_FILE,
        x_train=train_images,
        y_train=train_labels,
        x_test=test_images,
        y_test=test_labels,
    )

    return train_images, train_labels, test_images, test_labels


def load_fashion_mnist_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if FASHION_MNIST_CACHE_FILE.exists():
        with np.load(FASHION_MNIST_CACHE_FILE) as dataset:
            return (
                dataset["x_train"],
                dataset["y_train"],
                dataset["x_test"],
                dataset["y_test"],
            )

    dataset_dir = download_dataset_dir()
    return build_cache(dataset_dir)

