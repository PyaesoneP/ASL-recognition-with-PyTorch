"""Inference service: ASLPredictor and SentenceBuilder for the web API.

Self-contained inference module — does not import from src.inference
to avoid the mediapipe segfault in server environments.

Architecture compliance:
- ImagePreprocessor: Stage 4 (preprocessing)
- InferenceEngine/ASLPredictor: Stage 5 (classification)
- TemporalSmoother: Stage 6 (temporal smoothing)
- OutputFormatter/SentenceBuilder: Stage 7 (output formatting)
- ModelRegistry: Strategy pattern for model loading (ADR-005)
"""

import base64
import io
import os
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration (imported from centralized settings)
# ---------------------------------------------------------------------------

# Import from src.config.settings when available; fall back to defaults
try:
    import sys
    _sys_path_added = False
    _settings_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "src")
    if _settings_base not in sys.path:
        sys.path.insert(0, _settings_base)
        _sys_path_added = True

    from config.settings import (
        PREDICTION_DEFAULTS,
    )

    CONFIDENCE_THRESHOLD = PREDICTION_DEFAULTS["CONFIDENCE_THRESHOLD"]
    STABILITY_FRAMES = PREDICTION_DEFAULTS["STABILITY_FRAMES"]
    COOLDOWN_FRAMES = PREDICTION_DEFAULTS["COOLDOWN_FRAMES"]
    SMOOTHING_WINDOW = PREDICTION_DEFAULTS["SMOOTHING_WINDOW"]
    IMG_SIZE = PREDICTION_DEFAULTS["IMG_SIZE"]

    if _sys_path_added:
        sys.path.remove(_settings_base)
except ImportError:
    CONFIDENCE_THRESHOLD = 0.65
    STABILITY_FRAMES = 12
    COOLDOWN_FRAMES = 18
    SMOOTHING_WINDOW = 5
    IMG_SIZE = 224

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space',
]
NUM_CLASSES = 29

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Model definitions (must match training)
# ---------------------------------------------------------------------------

class CustomCNN(nn.Module):
    """Custom CNN architecture (must match training)."""

    def __init__(self, num_classes=29):
        super(CustomCNN, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# ModelRegistry — Strategy Pattern (ADR-005)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Strategy pattern for model loading.

    Implements ADR-005: config-driven model selection. The same loading
    interface works for any registered model type. MODEL_TYPE selects
    which strategy (architecture) to instantiate.
    """

    _registry = {
        "mobilenet_v2": models.mobilenet_v2,
        "resnet50": models.resnet50,
        "efficientnet_b0": models.efficientnet_b0,
    }

    @classmethod
    def get_model(cls, model_type: str, num_classes: int = NUM_CLASSES):
        """Instantiate a model architecture by type (strategy pattern)."""
        if model_type == "custom_cnn":
            return CustomCNN(num_classes)

        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        factory = cls._registry[model_type]
        model = factory(weights=None)

        # Replace final classifier layer for 29-class output
        if model_type == "mobilenet_v2":
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        elif model_type == "resnet50":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "efficientnet_b0":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        return model

    @classmethod
    def load_model(cls, model_path: str, model_type: str = "mobilenet_v2",
                   num_classes: int = NUM_CLASSES):
        """Load trained PyTorch model from disk.

        This is the public factory method — the entry point for the
        strategy pattern. Supports both raw state dicts and
        'model_state_dict' key formats.
        """
        model = cls.get_model(model_type, num_classes)

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(DEVICE)
        model.eval()

        return model


# ---------------------------------------------------------------------------
# ImagePreprocessor — Stage 4
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Transforms raw cropped image into model-ready tensor.

    Transform pipeline:
    1. BGR → RGB conversion (OpenCV)
    2. ToPILImage
    3. Resize to IMG_SIZE×IMG_SIZE
    4. ToTensor (values scaled to [0, 1])
    5. Normalize with ImageNet mean/std
    6. Add batch dimension
    7. Move to target device (CPU/CUDA)
    """

    def __init__(self, img_size: int = IMG_SIZE,
                 mean: list = IMAGENET_MEAN,
                 std: list = IMAGENET_STD):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def preprocess(self, image_bgr: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
        """Convert BGR numpy array to batched tensor on target device."""
        tensor = self.transform(image_bgr)
        return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# TemporalSmoother — Stage 6
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """Reduces prediction instability through multi-layer smoothing.

    Three smoothing layers:
    - Majority voting: Counter.most_common on history deque (5 frames)
    - Stability counting: Consecutive same-prediction counter (12 frames)
    - Cooldown: Post-commit lockout (18 frames)
    """

    def __init__(self, window_size: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)

    def smooth(self, top_idx: int, top_conf: float) -> Tuple[int, float]:
        """Apply majority voting over prediction history.

        Returns:
            (smoothed_idx, smoothed_confidence)
        """
        self.prediction_history.append((top_idx, top_conf))

        if len(self.prediction_history) >= 3:
            votes = [idx for idx, _ in self.prediction_history]
            most_common = max(set(votes), key=votes.count)
            vote_count = votes.count(most_common)
            smoothed_conf = top_conf * (vote_count / len(votes))

            if smoothed_conf >= self.confidence_threshold:
                top_idx = most_common
                top_conf = smoothed_conf

        return top_idx, top_conf


# ---------------------------------------------------------------------------
# InferenceEngine / ASLPredictor — Stage 5
# ---------------------------------------------------------------------------

class ASLPredictor:
    """Runs model forward pass with temporal smoothing via majority voting.

    Composed of ImagePreprocessor (Stage 4) and TemporalSmoother (Stage 6).
    This maintains backward compatibility with the original class name.
    """

    def __init__(self, model, model_type: str = "mobilenet_v2",
                 img_size: int = IMG_SIZE,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.model = model
        self.model_type = model_type

        # Stage 4: Preprocessing
        self.preprocessor = ImagePreprocessor(img_size=img_size)

        # Stage 6: Temporal smoothing
        self.smoother = TemporalSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image_bgr) -> Tuple[Optional[str], Optional[float], Optional[dict]]:
        """
        Run inference on a BGR image (numpy array).

        Returns:
            Tuple of (predicted_class, confidence, probabilities_dict)
            or (None, None, None) on error.
        """
        try:
            model_device = next(self.model.parameters()).device
            batch = self.preprocessor.preprocess(image_bgr, device=model_device)

            with torch.no_grad():
                output = self.model(batch)
                probs = torch.softmax(output, dim=1)
                probs = probs.squeeze().cpu().numpy()

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            # Apply temporal smoothing
            top_idx, top_conf = self.smoother.smooth(top_idx, top_conf)

            label = CLASS_NAMES[top_idx]
            prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                         for i in range(len(CLASS_NAMES))}

            return label, top_conf, prob_dict

        except Exception:
            return None, None, None


# ---------------------------------------------------------------------------
# OutputFormatter / SentenceBuilder — Stage 7
# ---------------------------------------------------------------------------

class SentenceBuilder:
    """Builds sentences with temporal smoothing: stability counting, cooldown,
    del/space handling.

    Acts as the OutputFormatter (Stage 7): accumulates recognized letters
    into coherent sentences. Handles special classes (space, del).
    Provides manual editing operations.
    """

    def __init__(self, stability_frames: int = STABILITY_FRAMES,
                 cooldown_frames: int = COOLDOWN_FRAMES):
        self.sentence = ""
        self.last_added_frame = -1
        self.stability_count = 0
        self.stability_threshold = stability_frames
        self.cooldown_frames = cooldown_frames
        self.current_letter = None
        self.frame_counter = 0

    def update(self, predicted_class: str, confidence: float = 0.0) -> Optional[str]:
        """
        Update sentence state with a new prediction.

        Returns:
            The letter added (if any), or None if nothing was added.
        """
        self.frame_counter += 1

        if predicted_class == "del":
            self.delete_last()
            return "del"
        elif predicted_class == "space":
            self.add_space()
            return "space"
        elif predicted_class == "nothing":
            return None

        if self.current_letter != predicted_class:
            if (self.current_letter is not None
                    and self.frame_counter - self.last_added_frame
                    >= self.cooldown_frames):
                self.current_letter = predicted_class
                self.stability_count = 1
            else:
                self.current_letter = predicted_class
                self.stability_count = 1
        else:
            self.stability_count += 1

        if self.stability_count >= self.stability_threshold:
            self._add_letter(predicted_class)
            self.current_letter = None
            self.stability_count = 0
            self.last_added_frame = self.frame_counter

        return None

    def _add_letter(self, letter: str):
        self.sentence += letter

    def add_space(self):
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        self.current_letter = None
        self.stability_count = 0
        self.last_added_frame = self.frame_counter

    def delete_last(self):
        if self.sentence:
            if self.sentence.endswith(" "):
                self.sentence = self.sentence[:-1]
            else:
                self.sentence = self.sentence.rstrip()
                if self.sentence:
                    self.sentence = self.sentence[:-1]
                else:
                    self.sentence = ""
        self.current_letter = None
        self.stability_count = 0

    def clear(self):
        self.sentence = ""
        self.current_letter = None
        self.stability_count = 0
        self.last_added_frame = -1
        self.frame_counter = 0

    def get_sentence(self) -> str:
        return self.sentence


# ---------------------------------------------------------------------------
# Inference Service (facade)
# ---------------------------------------------------------------------------

class InferenceService:
    """Facade that manages model loading, prediction, and sentence building
    per session.

    Orchestrates the pipeline stages:
    - ModelRegistry: loads model (ADR-005 strategy pattern)
    - ASLPredictor: Stage 5 (classification) with ImagePreprocessor (Stage 4)
                    and TemporalSmoother (Stage 6)
    - SentenceBuilder: Stage 7 (output formatting)
    """

    def __init__(self, model_path: Optional[str] = None,
                 model_type: Optional[str] = None,
                 img_size: int = IMG_SIZE,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 stability_frames: int = STABILITY_FRAMES,
                 cooldown_frames: int = COOLDOWN_FRAMES):
        self.model_path = model_path or "outputs/models/best_mobilenet_v2.pth"
        self.model_type = model_type or "mobilenet_v2"
        self.img_size = img_size
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames
        self.model = None
        self.predictor = None
        self._sentence_builders: dict = {}
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        # Use ModelRegistry (strategy pattern) to load model
        self.model = ModelRegistry.load_model(
            self.model_path, self.model_type
        )
        self.predictor = ASLPredictor(
            self.model,
            self.model_type,
            img_size=self.img_size,
            smoothing_window=self.smoothing_window,
            confidence_threshold=self.confidence_threshold,
        )
        self._initialized = True

    def _get_session_builder(self, session_id: str) -> SentenceBuilder:
        if session_id not in self._sentence_builders:
            self._sentence_builders[session_id] = SentenceBuilder(
                stability_frames=self.stability_frames,
                cooldown_frames=self.cooldown_frames,
            )
        return self._sentence_builders[session_id]

    def predict(self, image_bgr, session_id: str = "default"):
        if not self._initialized:
            self.initialize()
        return self.predictor.predict(image_bgr)

    def update_sentence(self, session_id: str, action: str = "predict",
                        prediction: Optional[str] = None) -> str:
        if not self._initialized:
            self.initialize()
        builder = self._get_session_builder(session_id)

        if action == "clear":
            builder.clear()
        elif action == "space":
            builder.add_space()
        elif action == "del":
            builder.delete_last()
        elif action == "predict" and prediction:
            builder.update(prediction)

        return builder.get_sentence()

    def get_sentence(self, session_id: str = "default") -> str:
        if not self._initialized:
            self.initialize()
        builder = self._get_session_builder(session_id)
        return builder.get_sentence()

    def decode_base64_image(self, image_b64: str) -> np.ndarray:
        """Decode base64-encoded JPEG image to BGR numpy array."""
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",")[1]
        image_data = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        arr = np.array(pil_image)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Backward compatibility wrapper
# ---------------------------------------------------------------------------

def load_model(model_path: str, model_type: str = "mobilenet_v2",
               num_classes: int = NUM_CLASSES):
    """Backward-compatible wrapper around ModelRegistry.load_model()."""
    return ModelRegistry.load_model(model_path, model_type, num_classes)
