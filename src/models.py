"""Shared model definitions and the ModelRegistry strategy (ADR-005).

Single source of truth for `CustomCNN` and the model factories, imported by both
the desktop pipeline (`src/inference`) and the web API (`api/services/predictor.py`).

Two robustness properties matter here:

1. **Torch is optional.** The ONNX-only API container installs `onnxruntime` but
   *not* PyTorch. Torch is imported under a guard so this module still imports
   there — `CustomCNN` becomes ``None`` and only the ONNX inference path runs.
2. **Dual sys.path layout.** The API puts ``.../src`` on ``sys.path`` (so
   ``config.settings`` is importable), while the desktop app / tests put the repo
   root on ``sys.path`` (so ``src.config.settings`` is importable). We try both.
"""

try:  # API context: ``.../src`` is on sys.path
    from config.settings import CLASS_NAMES, NUM_CLASSES
except ImportError:  # desktop / test context: repo root is on sys.path
    from src.config.settings import CLASS_NAMES, NUM_CLASSES


# ---------------------------------------------------------------------------
# CustomCNN — lightweight 4-block CNN (trained from scratch). torch-guarded so
# the ONNX-only container (no torch) still imports this module.
# ---------------------------------------------------------------------------

try:
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
# ModelRegistry — Strategy Pattern (ADR-005)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Strategy pattern for model architecture selection.

    Registers model factory functions keyed by model type string. The same
    interface works for any registered model backbone.

    Registered models:
        mobilenet_v2    — MobileNetV2 (ImageNet pretrained backbone)
        resnet50        — ResNet50 (ImageNet pretrained backbone)
        efficientnet_b0 — EfficientNet-B0 (ImageNet pretrained backbone)
        custom_cnn      — Custom 4-block CNN (trained from scratch)
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


# Factory functions import torch/torchvision lazily (inside the body) so merely
# *defining* and *registering* them needs no torch — only calling get_model()
# does. That keeps the registry populated even in the ONNX-only container.
def _register_mobilenet_v2(num_classes: int = NUM_CLASSES):
    import torch.nn as nn
    from torchvision import models as tv_models
    model = tv_models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _register_resnet50(num_classes: int = NUM_CLASSES):
    import torch.nn as nn
    from torchvision import models as tv_models
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _register_efficientnet_b0(num_classes: int = NUM_CLASSES):
    import torch.nn as nn
    from torchvision import models as tv_models
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


ModelRegistry.register("mobilenet_v2", _register_mobilenet_v2)
ModelRegistry.register("resnet50", _register_resnet50)
ModelRegistry.register("efficientnet_b0", _register_efficientnet_b0)
if CustomCNN is not None:
    ModelRegistry.register("custom_cnn", lambda num_classes=NUM_CLASSES: CustomCNN(num_classes))
