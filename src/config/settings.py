"""Default configuration settings for the ASL recognition project."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_MODEL_PATH = "outputs/models/best_mobilenet_v2.pth"
MODEL_TYPES = ["mobilenet_v2", "resnet50", "efficientnet_b0", "custom_cnn"]

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_TYPE = os.getenv("MODEL_TYPE", "mobilenet_v2")

DATASETS = {
    "train": "datasets/asl_alphabet_train/",
    "test": "datasets/asl_alphabet_test/",
    "organized": "datasets/asl_test_organized/",
    "combined": "datasets/combined_training/",
    "custom": "datasets/custom_dataset/",
}

PREDICTION_DEFAULTS = {
    "CONFIDENCE_THRESHOLD": float(os.getenv("CONFIDENCE_THRESHOLD", 0.65)),
    "STABILITY_FRAMES": int(os.getenv("STABILITY_FRAMES", 12)),
    "COOLDOWN_FRAMES": int(os.getenv("COOLDOWN_FRAMES", 18)),
    "SMOOTHING_WINDOW": int(os.getenv("SMOOTHING_WINDOW", 5)),
    "IMG_SIZE": int(os.getenv("IMG_SIZE", 224)),
}
