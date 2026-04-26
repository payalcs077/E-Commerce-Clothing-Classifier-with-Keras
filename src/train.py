import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
keras = tf.keras
layers = keras.layers
from sklearn.metrics import confusion_matrix

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONFUSION_MATRIX_FILE,
    EPOCHS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LABELS_FILE,
    LEARNING_RATE,
    MODEL_FILE,
    MODELS_DIR,
    OUTPUTS_DIR,
    PLOT_FILE,
    PREDICTION_PREVIEW_FILE,
    SEED,
    VALIDATION_SPLIT,
)
from fashion_mnist_kaggle import load_fashion_mnist_dataset


def load_and_prepare_data():
    x_train, y_train, x_test, y_test = load_fashion_mnist_dataset()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text("\n".join(CLASS_NAMES))


def plot_history(history):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_positions = range(len(CLASS_NAMES))
    plt.xticks(tick_positions, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_positions, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()


def save_prediction_preview(x_test, y_test, predictions):
    rng = np.random.default_rng(SEED)
    preview_indices = rng.choice(len(x_test), size=9, replace=False)

    plt.figure(figsize=(9, 9))
    for plot_position, sample_index in enumerate(preview_indices, start=1):
        predicted_label = CLASS_NAMES[int(np.argmax(predictions[sample_index]))]
        true_label = CLASS_NAMES[int(y_test[sample_index])]

        plt.subplot(3, 3, plot_position)
        plt.imshow(x_test[sample_index].squeeze(), cmap="gray")
        plt.title(f"T: {true_label}\nP: {predicted_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(PREDICTION_PREVIEW_FILE)
    plt.close()


def main():
    tf.keras.utils.set_random_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    save_labels()

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_FILE),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plot_confusion_matrix(y_test, predicted_labels)
    save_prediction_preview(x_test, y_test, predictions)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved labels to: {LABELS_FILE}")
    print(f"Saved training history to: {PLOT_FILE}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_FILE}")
    print(f"Saved prediction preview to: {PREDICTION_PREVIEW_FILE}")


if __name__ == "__main__":
    main()
