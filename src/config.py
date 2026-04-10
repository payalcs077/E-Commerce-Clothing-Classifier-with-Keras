from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
SEED = 42

FASHION_MNIST_DATASET_SLUG = "zalando-research/fashionmnist"
FASHION_MNIST_CACHE_FILE = CACHE_DIR / "fashion_mnist_kaggle.npz"

MODEL_FILE = MODELS_DIR / "fashion_mnist_classifier.keras"
LABELS_FILE = MODELS_DIR / "labels.txt"
PLOT_FILE = OUTPUTS_DIR / "training_history.png"
SAMPLE_GRID_FILE = OUTPUTS_DIR / "fashion_mnist_samples.png"
CONFUSION_MATRIX_FILE = OUTPUTS_DIR / "confusion_matrix.png"
PREDICTION_PREVIEW_FILE = OUTPUTS_DIR / "prediction_preview.png"
