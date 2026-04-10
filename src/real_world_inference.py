from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from real_world_config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    RAW_DIR,
    REAL_WORLD_LABELS_FILE,
    REAL_WORLD_MODEL_FILE,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def load_real_world_labels() -> list[str]:
    if not REAL_WORLD_LABELS_FILE.exists():
        raise FileNotFoundError(
            "Labels file not found. Train the real-world model first with src/train_real_world.py."
        )
    return REAL_WORLD_LABELS_FILE.read_text().splitlines()


def load_real_world_model():
    if not REAL_WORLD_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model file not found. Train the real-world model first with src/train_real_world.py."
        )
    return tf.keras.models.load_model(REAL_WORLD_MODEL_FILE)


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = tf.keras.utils.img_to_array(rgb_image)
    return np.expand_dims(image_array, axis=0)


def predict_pil_image(image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
    model = load_real_world_model()
    labels = load_real_world_labels()

    batch = preprocess_pil_image(image)
    probabilities = model.predict(batch, verbose=0)[0]

    ranked = sorted(
        zip(labels, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_score = ranked[0]
    return top_label, float(top_score), ranked


def predict_image_path(image_path: str | Path) -> tuple[str, float, list[tuple[str, float]]]:
    with Image.open(image_path) as image:
        return predict_pil_image(image)


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for file_path in directory.rglob("*") if file_path.is_file())


def dataset_status() -> dict[str, int]:
    return {
        "raw": count_images(RAW_DIR),
        "train": count_images(TRAIN_DIR),
        "val": count_images(VAL_DIR),
        "test": count_images(TEST_DIR),
    }

