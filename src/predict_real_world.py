import argparse
from pathlib import Path

from real_world_inference import predict_image_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label, confidence, ranked = predict_image_path(image_path)

    print(f"Predicted label: {label}")
    print(f"Confidence: {confidence:.2%}")
    print("Top predictions:")
    for ranked_label, score in ranked[:5]:
        print(f"  - {ranked_label}: {score:.2%}")


if __name__ == "__main__":
    main()

