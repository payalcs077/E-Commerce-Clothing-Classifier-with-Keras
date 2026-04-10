import random
import shutil
from pathlib import Path

from real_world_config import (
    IMAGE_EXTENSIONS,
    RAW_DIR,
    REAL_WORLD_DEFAULT_CLASSES,
    SEED,
    TEST_DIR,
    TEST_SPLIT,
    TRAIN_DIR,
    TRAIN_SPLIT,
    VAL_DIR,
    VALIDATION_SPLIT,
)


def ensure_raw_class_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in REAL_WORLD_DEFAULT_CLASSES:
        (RAW_DIR / class_name).mkdir(parents=True, exist_ok=True)


def validate_split_ratios() -> None:
    total = TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT
    if abs(total - 1.0) > 1e-9:
        raise ValueError("TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT must equal 1.0")


def clear_existing_split_data() -> None:
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for child in split_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def list_class_directories() -> list[Path]:
    return sorted(directory for directory in RAW_DIR.iterdir() if directory.is_dir())


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in class_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_split_files(class_name: str, files: list[Path], destination_root: Path) -> None:
    destination_dir = destination_root / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    for index, source_file in enumerate(files, start=1):
        destination_name = f"{class_name}_{index:05d}{source_file.suffix.lower()}"
        shutil.copy2(source_file, destination_dir / destination_name)


def split_class_images(class_name: str, images: list[Path]) -> None:
    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VALIDATION_SPLIT)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    copy_split_files(class_name, train_files, TRAIN_DIR)
    copy_split_files(class_name, val_files, VAL_DIR)
    copy_split_files(class_name, test_files, TEST_DIR)

    print(
        f"{class_name}: total={total}, train={len(train_files)}, "
        f"val={len(val_files)}, test={len(test_files)}"
    )


def main() -> None:
    random.seed(SEED)
    validate_split_ratios()
    ensure_raw_class_directories()
    clear_existing_split_data()

    class_directories = list_class_directories()
    total_images = 0

    for class_dir in class_directories:
        images = collect_images(class_dir)
        if not images:
            print(f"Skipping {class_dir.name}: no supported images found")
            continue

        random.shuffle(images)
        split_class_images(class_dir.name, images)
        total_images += len(images)

    if total_images == 0:
        print("No raw images found.")
        print(f"Add files to: {RAW_DIR}/<class_name>/")
        print(f"Expected classes: {', '.join(REAL_WORLD_DEFAULT_CLASSES)}")
        return

    print("Directory dataset prepared successfully.")


if __name__ == "__main__":
    main()

