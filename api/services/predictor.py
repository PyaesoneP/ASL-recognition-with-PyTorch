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
from collections import deque, OrderedDict
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

# ---------------------------------------------------------------------------
# Configuration (imported from centralized settings)
# ---------------------------------------------------------------------------

try:
    import sys
    _settings_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "src")
    if _settings_base not in sys.path:
        sys.path.insert(0, _settings_base)

    from config.settings import (
        PREDICTION_DEFAULTS,
    )

    CONFIDENCE_THRESHOLD = PREDICTION_DEFAULTS["CONFIDENCE_THRESHOLD"]
    STABILITY_FRAMES = PREDICTION_DEFAULTS["STABILITY_FRAMES"]
    COOLDOWN_FRAMES = PREDICTION_DEFAULTS["COOLDOWN_FRAMES"]
    SMOOTHING_WINDOW = PREDICTION_DEFAULTS["SMOOTHING_WINDOW"]
    IMG_SIZE = PREDICTION_DEFAULTS["IMG_SIZE"]

    if _settings_base in sys.path:
        sys.path.remove(_settings_base)
except ImportError:
    CONFIDENCE_THRESHOLD = 0.5
    STABILITY_FRAMES = 6
    COOLDOWN_FRAMES = 18
    SMOOTHING_WINDOW = 3
    IMG_SIZE = 224

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space',
]
NUM_CLASSES = 29

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Bounds on decoded images from public/unauthenticated endpoints (DoS guard).
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 6 * 1024 * 1024))   # 6 MB
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", 4096 * 4096))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS   # PIL raises DecompressionBombError past 2x

# ---------------------------------------------------------------------------
# CustomCNN — lightweight 4-block CNN (trained from scratch)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class CustomCNN(nn.Module):
        """Custom CNN architecture (must match training!).

        4 convolutional blocks (32->64->128->256 filters) with batch norm,
        ReLU, max pooling, and dropout. Global average pooling followed by
        a 3-layer classifier (256->512->256->num_classes).
        """

        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()

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

except ImportError:
    CustomCNN = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Inference Engine (ONNX Runtime)
# ---------------------------------------------------------------------------

class OnnxInferenceEngine:
    """
    High-performance inference engine using ONNX Runtime.
    Supports both CPU and GPU (via appropriate providers).
    """
    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.session = None
        self.input_name = None
        self._initialized = False
        self.initialize()

    def initialize(self):
        if self._initialized:
            return

        # Choose provider: CUDA if available, else CPU
        providers = ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._initialized = True
        except Exception as e:
            print(f"Error initializing ONNX session: {e}")
            raise

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs inference on the input tensor.
        input_tensor: numpy array of shape (1, 3, 224, 224)
        """
        return self.session.run(None, {self.input_name: input_tensor})[0]

# ---------------------------------------------------------------------------
# ImagePreprocessor — Stage 4
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Transforms raw cropped image into model-ready tensor (Numpy format)."""
    def __init__(self, img_size: int = IMG_SIZE,
                 mean: list = IMAGENET_MEAN,
                 std: list = IMAGENET_STD):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array(std, dtype=np.float32).reshape((3, 1, 1))

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR numpy array to batched NCHW tensor for ONNX.
        """
        # 1. BGR to RGB
        image_rgb = image_bgr[:, :, ::-1].copy()

        # 2. Resize
        image_resized = Image.fromarray(image_rgb).resize((self.img_size, self.img_size))
        image_arr = np.array(image_resized).astype(np.float32)

        # 3. HWC to CHW and normalize per-channel
        image_arr = image_arr.transpose(2, 0, 1)  # (3, 224, 224)
        image_arr /= 255.0
        image_arr = (image_arr - self.mean) / self.std

        # 4. Add batch dimension
        image_arr = np.expand_dims(image_arr, axis=0)

        return image_arr

# ---------------------------------------------------------------------------
# TemporalSmoother — Stage 6
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """Reduces prediction instability through multi-layer smoothing."""
    def __init__(self, window_size: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)

    def smooth(self, top_idx: int, top_conf: float) -> Tuple[int, float]:
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
# ASLPredictor — Stage 5
# ---------------------------------------------------------------------------

class ASLPredictor:
    """Runs model forward pass with temporal smoothing."""
    def __init__(self, engine: OnnxInferenceEngine,
                 img_size: int = IMG_SIZE,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.engine = engine
        self.img_size = img_size
        self.smoother = TemporalSmoother(
            window_size=smoothing_window,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image_bgr: np.ndarray, smoother: "TemporalSmoother" = None) -> Tuple[Optional[str], Optional[float], Optional[dict]]:
        try:
            # Stage 4: Preprocessing
            preprocessor = ImagePreprocessor(img_size=self.img_size)
            input_tensor = preprocessor.preprocess(image_bgr)

            # Stage 5: Inference
            output = self.engine.run(input_tensor)
            probs = np.exp(output) / np.sum(np.exp(output), axis=1) # Softmax
            probs = probs[0] # Remove batch dim

            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            # Stage 6: Temporal smoothing (per-session smoother when provided)
            sm = smoother if smoother is not None else self.smoother
            top_idx, top_conf = sm.smooth(top_idx, top_conf)

            label = CLASS_NAMES[top_idx]
            prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                         for i in range(len(CLASS_NAMES))}

            return label, top_conf, prob_dict

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None

# ---------------------------------------------------------------------------
# OutputFormatter / SentenceBuilder — Stage 7
# ---------------------------------------------------------------------------

class SentenceBuilder:
    """Builds sentences with temporal smoothing: stability counting,
    cooldown, del/space handling.
    """
    def __init__(self, stability_frames: int = STABILITY_FRAMES,
                 cooldown_frames: int = COOLDOWN_FRAMES,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.sentence = ""
        self.last_added_frame = -1
        self.stability_count = 0
        self.stability_threshold = stability_frames
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = confidence_threshold
        self.current_letter = None
        self.frame_counter = 0

    def update(self, predicted_class: str, confidence: float = 1.0) -> Optional[str]:
        self.frame_counter += 1

        if predicted_class == "del":
            self.delete_last()
            return "del"
        elif predicted_class == "space":
            self.add_space()
            return "space"
        elif predicted_class == "nothing":
            return None

        # Confidence gate — makes CONFIDENCE_THRESHOLD meaningful. Predictions
        # below threshold don't advance stability (default 1.0 so callers that
        # don't supply a confidence still commit).
        if confidence < self.confidence_threshold:
            return None

        # Stability counting: a sign must be held for stability_threshold
        # consecutive frames before it commits. (The old cooldown branch here
        # was dead code — both arms were identical — so it's removed.)
        if self.current_letter != predicted_class:
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
        self.last_added_frame = self.frame_counter

    def clear(self):
        self.sentence = ""
        self.current_letter = None
        self.stability_count = 0
        self.last_added_frame = -1
        self.frame_counter = 0

    def get_sentence(self) -> str:
        return self.sentence

# ---------------------------------------------------------------------------
# ModelRegistry — Strategy Pattern (ADR-005)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Strategy pattern for model architecture selection.

    Registers model factory functions keyed by model type string.
    The same interface works for any registered model backbone.

    Registered models:
        mobilenet_v2  — MobileNetV2 (ImageNet pretrained backbone)
        resnet50      — ResNet50 (ImageNet pretrained backbone)
        efficientnet_b0 — EfficientNet-B0 (ImageNet pretrained backbone)
        custom_cnn    — Custom 4-block CNN (trained from scratch)
    """

    _registry: dict = {}

    @classmethod
    def register(cls, model_type: str, factory):
        """Register a model factory function for the given model type."""
        cls._registry[model_type] = factory

    @classmethod
    def get_model(cls, model_type: str, num_classes: int = NUM_CLASSES):
        """Instantiate a model by registered type.

        Args:
            model_type: One of 'mobilenet_v2', 'resnet50', 'efficientnet_b0', 'custom_cnn'.
            num_classes: Number of output classes (default 29).

        Returns:
            An nn.Module instance ready for weights loading.

        Raises:
            ValueError: If model_type is not registered.
        """
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Registered types: {list(cls._registry.keys())}"
            )
        return cls._registry[model_type](num_classes=num_classes)

    @classmethod
    def list_models(cls):
        """Return list of registered model types."""
        return list(cls._registry.keys())


def _register_mobilenet_v2(num_classes: int = NUM_CLASSES):
    from torchvision import models as tv_models
    model = tv_models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _register_resnet50(num_classes: int = NUM_CLASSES):
    from torchvision import models as tv_models
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _register_efficientnet_b0(num_classes: int = NUM_CLASSES):
    from torchvision import models as tv_models
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Register all four model backbones (only if torch/torchvision available)
try:
    ModelRegistry.register("mobilenet_v2", _register_mobilenet_v2)
    ModelRegistry.register("resnet50", _register_resnet50)
    ModelRegistry.register("efficientnet_b0", _register_efficientnet_b0)
    if CustomCNN is not None:
        ModelRegistry.register("custom_cnn", lambda num_classes=NUM_CLASSES: CustomCNN(num_classes))
except (ImportError, NameError):
    pass  # PyTorch-based models unavailable; ONNX inference still works


# ---------------------------------------------------------------------------
# Inference Service (facade)
# ---------------------------------------------------------------------------

class InferenceService:
    """Facade that manages model loading, prediction, and sentence building
    per session.
    """
    def __init__(self, model_path: Optional[str] = None,
                 model_type: Optional[str] = None,
                 img_size: int = IMG_SIZE,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 stability_frames: int = STABILITY_FRAMES,
                 cooldown_frames: int = COOLDOWN_FRAMES):
        self.model_path = model_path or "outputs/models/best_mobilenet_v2.onnx"
        self.model_type = model_type or "mobilenet_v2"
        self.img_size = img_size
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames
        self.engine = None
        # Per-session state (SentenceBuilder + TemporalSmoother), LRU-capped so
        # unbounded client-supplied session_ids can't exhaust memory.
        self._sessions: "OrderedDict[str, dict]" = OrderedDict()
        self._max_sessions = int(os.getenv("MAX_SESSIONS", 512))
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        self.engine = OnnxInferenceEngine(self.model_path, self.model_type)
        self.predictor = ASLPredictor(
            self.engine,
            img_size=self.img_size,
            smoothing_window=self.smoothing_window,
            confidence_threshold=self.confidence_threshold,
        )
        self._initialized = True

    def predict(self, image_bgr, session_id: str = "default"):
        if not self._initialized:
            self.initialize()
        session = self._get_session(session_id)
        return self.predictor.predict(image_bgr, smoother=session["smoother"])

    def update_sentence(self, session_id: str, action: str = "predict",
                        prediction: Optional[str] = None,
                        confidence: float = 1.0) -> str:
        if not self._initialized:
            self.initialize()
        builder = self._get_session(session_id)["builder"]
        if action == "clear":
            builder.clear()
        elif action == "space":
            builder.add_space()
        elif action == "del":
            builder.delete_last()
        elif action == "predict" and prediction:
            builder.update(prediction, confidence)
        return builder.get_sentence()

    def get_sentence(self, session_id: str = "default") -> str:
        """Retrieve the current sentence for a given session."""
        if not self._initialized:
            self.initialize()
        return self._get_session(session_id)["builder"].get_sentence()

    def _get_session(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            session = {
                "builder": SentenceBuilder(),
                "smoother": TemporalSmoother(
                    window_size=self.smoothing_window,
                    confidence_threshold=self.confidence_threshold,
                ),
            }
            self._sessions[session_id] = session
            # Evict least-recently-used sessions past the cap.
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return session

    def remove_session(self, session_id: str) -> None:
        """Drop a session's state (called when a WebSocket disconnects)."""
        self._sessions.pop(session_id, None)

    def decode_base64_image(self, base64_str: str) -> np.ndarray:
        if isinstance(base64_str, str):
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            # Reject oversized payloads before/after decode (base64 is ~4/3 of bytes).
            if len(base64_str) > MAX_IMAGE_BYTES * 4 // 3 + 4:
                raise ValueError("image payload too large")
            image_bytes = base64.b64decode(base64_str)
            if len(image_bytes) > MAX_IMAGE_BYTES:
                raise ValueError("image payload too large")
            image = Image.open(io.BytesIO(image_bytes))
            w, h = image.size
            if w <= 0 or h <= 0 or w * h > MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions out of bounds")
            image = image.convert('RGB')
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return base64_str

