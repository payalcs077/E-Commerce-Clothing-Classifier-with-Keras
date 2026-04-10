import argparse

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, MODEL_FILE
from fashion_mnist_kaggle import load_fashion_mnist_dataset


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Saved model not found. Run src/train.py first.")
    return tf.keras.models.load_model(MODEL_FILE)


def load_test_dataset():
    _, _, x_test, y_test = load_fashion_mnist_dataset()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def predict_by_index(sample_index):
    model = load_model()
    x_test, y_test = load_test_dataset()

    if sample_index < 0 or sample_index >= len(x_test):
        raise IndexError(f"Index must be between 0 and {len(x_test) - 1}")

    sample = np.expand_dims(x_test[sample_index], axis=0)
    prediction = model.predict(sample, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    true_index = int(y_test[sample_index])

    print(f"Sample index: {sample_index}")
    print(f"True label: {CLASS_NAMES[true_index]}")
    print(f"Predicted label: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {prediction[predicted_index]:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Fashion MNIST test sample index to classify",
    )
    args = parser.parse_args()
    predict_by_index(args.index)


if __name__ == "__main__":
    main()
