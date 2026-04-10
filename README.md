# Clothing Classifier with Keras

This repository now contains two workflows:

1. A complete `Fashion MNIST` baseline you can run immediately
2. A real-world e-commerce image pipeline that trains from your own folder-based dataset and serves predictions through a small Streamlit UI

## Project Structure

```text
Clothing_classifier/
├── data/
│   ├── raw/
│   │   ├── dress/
│   │   ├── jacket/
│   │   ├── jeans/
│   │   ├── shoes/
│   │   └── tshirt/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
├── outputs/
├── src/
│   ├── app_real_world.py
│   ├── config.py
│   ├── fashion_mnist_kaggle.py
│   ├── import_real_world_kaggle.py
│   ├── predict.py
│   ├── predict_real_world.py
│   ├── prepare_data.py
│   ├── prepare_real_data.py
│   ├── real_world_config.py
│   ├── real_world_inference.py
│   ├── train.py
│   └── train_real_world.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment

Use Python 3.11 on this machine:

```bash
cd /Users/payalmac/Developer/Code/Clothing_classifier
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow 1: Fashion MNIST Baseline

This path is fully runnable without collecting any images.
The dataset is downloaded with `kagglehub` from `zalando-research/fashionmnist` and cached locally as a compressed NumPy file under `data/cache/`.

Commands:

```bash
python src/prepare_data.py
python src/train.py
python src/predict.py --index 0
```

Artifacts:

- `models/fashion_mnist_classifier.keras`
- `models/labels.txt`
- `outputs/fashion_mnist_samples.png`
- `outputs/training_history.png`
- `outputs/confusion_matrix.png`
- `outputs/prediction_preview.png`

## Workflow 2: Real E-Commerce Product Images

This path can now bootstrap itself from a Kaggle product-image dataset and then train on a balanced subset.

Current starter classes:

- `dress`
- `jacket`
- `jeans`
- `shoes`
- `tshirt`

### Step 1: Import a real product-image dataset

Run:

```bash
python src/import_real_world_kaggle.py
```

This downloads `paramaggarwal/fashion-product-images-small`, maps its catalog into:

- `dress`
- `jacket`
- `jeans`
- `shoes`
- `tshirt`

and imports a balanced subset into `data/raw/`.

### Step 2: Build train/validation/test splits

```bash
python src/prepare_real_data.py
```

This creates the split directories under `data/train`, `data/val`, and `data/test`.

### Step 3: Train the transfer-learning model

```bash
python src/train_real_world.py
```

This script uses `MobileNetV2` with ImageNet weights and saves:

- `models/real_world_clothing_classifier.keras`
- `models/real_world_labels.txt`
- `outputs/real_world_training_history.png`
- `outputs/real_world_confusion_matrix.png`

### Step 4: Predict on one image from the terminal

```bash
python src/predict_real_world.py --image /absolute/path/to/image.jpg
```

### Step 5: Launch the local web UI

```bash
streamlit run src/app_real_world.py
```

The UI uploads a product image, runs inference, and shows ranked class probabilities.

## Notes

- The Fashion MNIST path is already trained in this workspace
- The real-world path can now import a dataset automatically and then train without manual image collection
- The real-world model uses RGB images resized to `224x224`
