import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from real_world_config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    MODELS_DIR,
    OUTPUTS_DIR,
    REAL_WORLD_CONFUSION_MATRIX_FILE,
    REAL_WORLD_HISTORY_FILE,
    REAL_WORLD_LABELS_FILE,
    REAL_WORLD_MODEL_FILE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

AUTOTUNE = tf.data.AUTOTUNE


def count_class_directories(directory) -> int:
    return sum(1 for path in directory.iterdir() if path.is_dir()) if directory.exists() else 0


def validate_prepared_dataset() -> None:
    if not TRAIN_DIR.exists() or count_class_directories(TRAIN_DIR) == 0:
        raise FileNotFoundError(
            "Prepared training data not found. Add images to data/raw and run src/prepare_real_data.py."
        )

    for split_dir in [VAL_DIR, TEST_DIR]:
        if not split_dir.exists() or count_class_directories(split_dir) == 0:
            raise FileNotFoundError(
                f"Prepared split missing: {split_dir}. Run src/prepare_real_data.py."
            )


def load_dataset(directory, shuffle):
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )


def load_datasets():
    validate_prepared_dataset()

    train_ds = load_dataset(TRAIN_DIR, shuffle=True)
    val_ds = load_dataset(VAL_DIR, shuffle=False)
    test_ds = load_dataset(TEST_DIR, shuffle=False)

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(class_names: list[str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REAL_WORLD_LABELS_FILE.write_text("\n".join(class_names))


def plot_history(history) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Real-World Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Real-World Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(REAL_WORLD_HISTORY_FILE)
    plt.close()


def plot_confusion_matrix(true_labels, predicted_labels, class_names: list[str]) -> None:
    matrix = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Real-World Confusion Matrix")
    plt.colorbar()
    tick_positions = range(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45, ha="right")
    plt.yticks(tick_positions, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(REAL_WORLD_CONFUSION_MATRIX_FILE)
    plt.close()


def collect_true_labels(dataset) -> np.ndarray:
    return np.concatenate([labels.numpy() for _, labels in dataset], axis=0)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets()
    save_labels(class_names)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=REAL_WORLD_MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = collect_true_labels(test_ds)
    plot_confusion_matrix(true_labels, predicted_labels, class_names)

    print(f"Real-world test loss: {test_loss:.4f}")
    print(f"Real-world test accuracy: {test_accuracy:.4f}")
    print(f"Saved model to: {REAL_WORLD_MODEL_FILE}")
    print(f"Saved labels to: {REAL_WORLD_LABELS_FILE}")
    print(f"Saved history plot to: {REAL_WORLD_HISTORY_FILE}")
    print(f"Saved confusion matrix to: {REAL_WORLD_CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    main()

