import random
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from real_world_config import (
    OUTPUTS_DIR,
    RAW_DIR,
    REAL_WORLD_DEFAULT_CLASSES,
    REAL_WORLD_IMPORT_ARTICLE_TYPES,
    REAL_WORLD_IMPORT_LIMIT_PER_CLASS,
    REAL_WORLD_IMPORT_MANIFEST_FILE,
    REAL_WORLD_SOURCE_DATASET_SLUG,
    SEED,
)


def download_dataset_dir() -> Path:
    return Path(kagglehub.dataset_download(REAL_WORLD_SOURCE_DATASET_SLUG))


def load_styles_dataframe(dataset_dir: Path) -> pd.DataFrame:
    styles = pd.read_csv(dataset_dir / "styles.csv", on_bad_lines="skip")
    styles["id"] = styles["id"].astype(str)
    return styles


def image_path_for_id(images_dir: Path, image_id: str) -> Path:
    return images_dir / f"{image_id}.jpg"


def reset_raw_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in REAL_WORLD_DEFAULT_CLASSES:
        class_dir = RAW_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for child in class_dir.iterdir():
            if child.is_file():
                child.unlink()


def select_class_rows(styles: pd.DataFrame, images_dir: Path, class_name: str) -> pd.DataFrame:
    article_types = REAL_WORLD_IMPORT_ARTICLE_TYPES[class_name]
    filtered = styles.loc[styles["articleType"].isin(article_types)].copy()
    image_ids = filtered["id"].astype(str)
    filtered["image_path"] = image_ids.map(lambda image_id: image_path_for_id(images_dir, str(image_id)))
    filtered = filtered.loc[filtered["image_path"].map(Path.exists)].copy()
    filtered = filtered.drop_duplicates(subset=["id"])
    return filtered


def copy_class_images(class_name: str, rows: pd.DataFrame) -> int:
    target_dir = RAW_DIR / class_name
    copied = 0

    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        source_file = row["image_path"]
        destination_file = target_dir / f"{class_name}_{index:05d}{source_file.suffix.lower()}"
        shutil.copy2(source_file, destination_file)
        copied += 1

    return copied


def main() -> None:
    random.seed(SEED)
    dataset_dir = download_dataset_dir()
    styles = load_styles_dataframe(dataset_dir)
    images_dir = dataset_dir / "images"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_raw_directories()

    manifest_rows: list[dict[str, object]] = []

    for class_name in REAL_WORLD_DEFAULT_CLASSES:
        eligible_rows = select_class_rows(styles, images_dir, class_name)
        selected_rows = eligible_rows.sample(
            n=min(REAL_WORLD_IMPORT_LIMIT_PER_CLASS, len(eligible_rows)),
            random_state=SEED,
        ).sort_values("id")

        copied_count = copy_class_images(class_name, selected_rows)
        manifest_rows.append(
            {
                "class_name": class_name,
                "article_types": ", ".join(sorted(REAL_WORLD_IMPORT_ARTICLE_TYPES[class_name])),
                "available_images": int(len(eligible_rows)),
                "imported_images": copied_count,
            }
        )
        print(
            f"{class_name}: available={len(eligible_rows)}, imported={copied_count}, "
            f"article_types={sorted(REAL_WORLD_IMPORT_ARTICLE_TYPES[class_name])}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(REAL_WORLD_IMPORT_MANIFEST_FILE, index=False)
    print(f"Saved import manifest to: {REAL_WORLD_IMPORT_MANIFEST_FILE}")


if __name__ == "__main__":
    main()
