import math

import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_NAMES, OUTPUTS_DIR, SAMPLE_GRID_FILE, SEED
from fashion_mnist_kaggle import load_fashion_mnist_dataset


def load_dataset():
    return load_fashion_mnist_dataset()


def print_dataset_summary(y_train, y_test):
    print("Dataset summary")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    print("\nTraining class distribution")
    for class_index, class_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == class_index))
        print(f"{class_name}: {count}")

    print("\nTest class distribution")
    for class_index, class_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_test == class_index))
        print(f"{class_name}: {count}")


def save_sample_grid(x_train, y_train):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cols = 5
    rows = math.ceil(len(CLASS_NAMES) / cols) * 2
    plt.figure(figsize=(12, 8))

    for plot_index, class_index in enumerate(range(len(CLASS_NAMES)), start=1):
        class_examples = np.where(y_train == class_index)[0]
        chosen_index = int(rng.choice(class_examples))

        plt.subplot(rows, cols, plot_index)
        plt.imshow(x_train[chosen_index], cmap="gray")
        plt.title(CLASS_NAMES[class_index])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(SAMPLE_GRID_FILE)
    plt.close()
    print(f"Saved sample grid to: {SAMPLE_GRID_FILE}")


def main():
    x_train, y_train, x_test, y_test = load_dataset()
    print_dataset_summary(y_train, y_test)
    save_sample_grid(x_train, y_train)


if __name__ == "__main__":
    main()
