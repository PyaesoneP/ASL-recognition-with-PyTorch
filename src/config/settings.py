"""Default configuration settings for the ASL recognition project."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_PATH = "outputs/models/best_mobilenet_v2.pth"
MODEL_TYPES = ["mobilenet_v2", "resnet50", "efficientnet_b0", "custom_cnn"]

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_TYPE = os.getenv("MODEL_TYPE", "mobilenet_v2")

# ASL class labels — the 29 Kaggle ASL Alphabet classes (A-Z, del, nothing,
# space). Single source of truth shared by src/models.py, the desktop pipeline
# (src/inference), and the ONNX API (api/services/predictor.py).
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space',
]
NUM_CLASSES = len(CLASS_NAMES)

DATASETS = {
    "train": "datasets/asl_alphabet_train/",
    "test": "datasets/asl_alphabet_test/",
    "organized": "datasets/asl_test_organized/",
    "combined": "datasets/combined_training/",
    "cropped": "datasets/combined_cropped/",   # MediaPipe hand crops (train/serve parity)
    "custom": "datasets/custom_dataset/",
}

PREDICTION_DEFAULTS = {
    "CONFIDENCE_THRESHOLD": float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
    "STABILITY_FRAMES": int(os.getenv("STABILITY_FRAMES", 6)),
    "COOLDOWN_FRAMES": int(os.getenv("COOLDOWN_FRAMES", 18)),
    "SMOOTHING_WINDOW": int(os.getenv("SMOOTHING_WINDOW", 3)),
    "IMG_SIZE": int(os.getenv("IMG_SIZE", 224)),
}
