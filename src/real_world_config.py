from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

REAL_WORLD_DEFAULT_CLASSES = [
    "dress",
    "jacket",
    "jeans",
    "shoes",
    "tshirt",
]

REAL_WORLD_SOURCE_DATASET_SLUG = "paramaggarwal/fashion-product-images-small"

REAL_WORLD_IMPORT_ARTICLE_TYPES = {
    "dress": {"Dresses"},
    "jacket": {"Jackets"},
    "jeans": {"Jeans"},
    "shoes": {
        "Casual Shoes",
        "Flats",
        "Flip Flops",
        "Formal Shoes",
        "Heels",
        "Sandals",
        "Sports Shoes",
    },
    "tshirt": {"Tshirts"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
TRAIN_SPLIT = 0.70
SEED = 42
REAL_WORLD_IMPORT_LIMIT_PER_CLASS = 250

REAL_WORLD_MODEL_FILE = MODELS_DIR / "real_world_clothing_classifier.keras"
REAL_WORLD_LABELS_FILE = MODELS_DIR / "real_world_labels.txt"
REAL_WORLD_HISTORY_FILE = OUTPUTS_DIR / "real_world_training_history.png"
REAL_WORLD_CONFUSION_MATRIX_FILE = OUTPUTS_DIR / "real_world_confusion_matrix.png"
REAL_WORLD_IMPORT_MANIFEST_FILE = OUTPUTS_DIR / "real_world_import_manifest.csv"
