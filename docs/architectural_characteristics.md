# Architectural Characteristics — ASL Recognition System

## 1. System Overview

Real-time American Sign Language (ASL) recognition system that captures hand gestures via webcam, classifies them using deep learning models, and outputs recognized letters and sentences. Designed for local inference with a path to live web application deployment.

### Pipeline Stages

```
Webcam Feed → Hand Detection (MediaPipe) → ROI Crop → CNN Inference → Temporal Smoothing → Output
```

| Stage | Component | Responsibility |
|-------|-----------|----------------|
| Data Collection | `src/scripts/capture_asl_images.py` | Interactive and batch image capture for dataset building |
| Training | Jupyter notebook (`notebooks/ASL_PyTorch_Complete.ipynb`) | Model training with augmentation, validation, and metrics |
| Inference (local) | `src/inference/__init__.py` | Desktop webcam gesture-to-text pipeline (MediaPipe + PyTorch) |
| Inference (web API) | `api/services/predictor.py` | ONNX Runtime inference service (self-contained, server-safe) |
| Configuration | `src/config/settings.py` | Centralized model paths, dataset paths, prediction parameters |

## 2. Core Non-Functional Requirements

### 2.1 Performance

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Inference latency | <33ms per frame | Required to sustain 30 FPS real-time operation |
| Frame rate | 30 FPS | Webcam capture and processing rate |
| Model size | <25M parameters | MobileNetV2 (~3.5M) preferred as default; ResNet50 (~25M) as accuracy option |
| Memory footprint | <2GB RAM | Suitable for consumer-grade hardware without dedicated GPU |
| GPU acceleration | CUDA when available | Automatic device selection; graceful CPU fallback |

### 2.2 Usability

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Real-time feedback | Letter predictions overlaid on video feed with confidence scores | Immediate visual confirmation of recognized gestures |
| Temporal stability | Majority voting over 5-frame window | Prevents prediction flicker and jitter |
| Confirmation threshold | 12 consecutive frames (~0.4s at 30fps) | Balances responsiveness with accuracy |
| Cooldown period | 18-frame cooldown between letter additions | Prevents double-counting of held gestures |
| User controls | Keyboard-driven: quit, clear, space, backspace, screenshot, reset | Full editing control without leaving inference mode |
| Dual modes | MediaPipe auto-detect and simple manual ROI | Graceful degradation when MediaPipe unavailable |

### 2.3 Accuracy & Reliability

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Confidence threshold | 65% minimum to accept a prediction | Filters low-confidence noise while maintaining throughput |
| Class coverage | 29 classes (A-Z excluding motion-required J, Z; plus del, nothing, space) | Covers static alphabet signs |
| Model options | 4 interchangeable backbones: MobileNetV2, ResNet50, EfficientNet-B0, Custom CNN | Trade-off flexibility between speed and accuracy |
| Swapability | Model type selected via config; inference transforms match training transforms | Guarantees consistent behavior across model swaps |
| Prediction smoothing | Majority voting + stability counting + cooldown | Three-layer defense against erratic predictions |

### 2.4 Extensibility

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Model hot-swap | Change `MODEL_TYPE` and `MODEL_PATH` in config to switch backbones | No code changes required to experiment with architectures |
| Dataset modularity | Multiple dataset sources defined in config; add new paths without code changes | Supports incremental dataset growth and A/B testing |
| Pipeline stage decoupling | Hand detection, cropping, inference, and smoothing are independent components | Each stage can be replaced or enhanced independently |
| Two-handed signs | Architecture supports extending beyond single-hand ROI | Foundation for future J, Z, and number recognition |
| Web deployment path | Inference logic separable from webcam capture for API service | Clean boundary between capture and classification |

### 2.5 Security & Privacy

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Data handling | All image processing local; no external data transmission | User privacy preserved during inference |
| Model protection | `.pth` files stored locally in `outputs/models/` | Trained models remain under user control |
| Webcam access | Direct device access only; no streaming to external services | Minimizes attack surface |
| Web deployment auth | Planned API authentication for cloud deployment | Prevents unauthorized model access when deployed |
| Captured data retention | User-controlled; captured images stored locally | No automatic upload or cloud sync |

### 2.6 Portability

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Primary platform | Windows-native with OpenCV webcam access | Development and primary deployment target |
| WSL2 compatibility | `--simple` flag fallback mode | Supports Linux development environments |
| Web deployment | Inference logic designed for containerization and API exposure | Path to browser-based access |
| Mobile deployment | Model architectures (MobileNetV2, EfficientNet) are mobile-friendly | Future conversion to TFLite or CoreML feasible |
| Hardware constraints | CPU inference supported; GPU optional | Runs on consumer laptops without dedicated GPU |

### 2.7 Maintainability

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Config-driven design | All paths, thresholds, and model selection via `src/config/settings.py` | Centralized control reduces refactoring surface |
| Modular components | Classes for HandDetector, ASLPredictor, SentenceBuilder, UIRenderer | Clear ownership and testability per component |
| Transform consistency | Inference transforms defined once, shared with training | Prevents train/inference mismatch bugs |
| Metrics export | Training metrics saved to `outputs/metrics/` | Enables post-hoc analysis without retraining |

## 3. Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Deep learning framework | PyTorch | Model definition, training, inference |
| Hand detection | MediaPipe | Real-time hand landmark detection |
| Computer vision | OpenCV | Camera access, image processing, UI rendering |
| Numerical computing | NumPy | Array operations, probability handling |
| Model architectures | Transfer learning (ImageNet pretrained) + custom CNN | Classification backbones |
| Training environment | Jupyter Notebook | Iterative model development |
| Target deployment | Live web application | End-user delivery channel |

## 4. Data Architecture

### Dataset Organization

```
datasets/
  asl_alphabet_train/     ← Primary training data
  asl_alphabet_test/      ← Primary test data
  asl_test_organized/     ← Additional organized test data
  combined_training/      ← Merged training dataset
  custom_dataset/         ← User-captured images
```

### Data Flow

1. **Collection**: Webcam captures ROI-cropped images per class (target: 100 images/class)
2. **Augmentation**: Random horizontal flip, color jitter, rotation, and normalization during training
3. **Training**: Combined dataset with oversampling for class balance
4. **Validation**: Separate test split for accuracy metrics
5. **Inference**: Live frames processed with same transforms as training

### Image Specs

- Input size: 224×224 pixels
- Format: RGB, normalized with ImageNet mean/std
- ROI: Variable crop with 25% padding around detected hand (simple mode: fixed 300×300)

## 5. Model Architecture

### Available Models

| Model | Type | Parameters | Use Case |
|-------|------|-----------|----------|
| MobileNetV2 | Transfer learning | ~3.5M | Default — best speed/accuracy trade-off |
| ResNet50 | Transfer learning | ~25M | Higher accuracy, higher latency |
| EfficientNet-B0 | Transfer learning | ~5M | Lightweight alternative |
| Custom CNN | From scratch | ~1M | Minimal footprint, lower accuracy |

### Training Strategy

- Transfer learning from ImageNet pretrained weights
- Custom classifier head replaces final layer (29 classes)
- Class-balanced sampling via weighted oversampling
- Training metrics: accuracy, loss curves, confusion matrices saved to `outputs/metrics/`

## 6. Inference Pipeline

### Frame Processing

```
Raw Frame → MediaPipe Hand Detection → Landmark-based ROI Crop (25% padding)
→ Square Resize → Center Crop → 224×224 → Tensor → Normalize → Model Forward Pass
→ Softmax Probabilities → Confidence Check → Temporal Smoothing → Letter Output
```

### Temporal Smoothing

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Smoothing window | 5 frames | Majority voting over recent predictions |
| Stability frames | 12 frames | Consecutive same prediction to commit letter |
| Cooldown frames | 18 frames | Minimum interval between letter additions |
| Confidence threshold | 0.65 | Minimum probability to accept prediction |

## 7. Deployment Considerations

### Current State

- Local inference on Windows with direct webcam access
- Model files stored in `outputs/models/`
- Configuration-driven model selection

### Web Application Path

- Inference logic containerized in `api/` (FastAPI + ONNX Runtime)
- Webcam capture moves to client-side (browser MediaPipe Hands JS)
- Model served via ONNX Runtime (optimized CPU inference)
- Temporal smoothing preserved server-side for consistent behavior
- Single-container deployment: API serves frontend via `StaticFiles` (Render-compatible)
- Docker Compose: separate `api` and `frontend` (nginx) services

## 8. Future Scope

- Two-handed sign recognition (J, Z, numbers)
- Continuous gesture sequences beyond single letters
- Multi-user support
- Mobile deployment
- Model retraining pipeline automation
