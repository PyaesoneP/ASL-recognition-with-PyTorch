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
| Cropping | `crop_dataset.py` | MediaPipe hand-crop preprocessing for train/serve parity (ADR-007) |
| Training | `train_and_export.py` (exploration: `notebooks/ASL_PyTorch_Complete.ipynb`) | Train MobileNetV2 on hand crops, export ONNX |
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
| Confidence gate | Commits require ≥ `CONFIDENCE_THRESHOLD` | Filters low-confidence predictions before they commit |
| User controls | Keyboard-driven: quit, clear, space, backspace, screenshot, reset (desktop); clear/space/backspace (web) | Full editing control without leaving inference mode |
| Dual modes | MediaPipe auto-detect and simple manual ROI | Graceful degradation when MediaPipe unavailable |

### 2.3 Accuracy & Reliability

| Characteristic | Target | Rationale |
|---------------|--------|-----------|
| Confidence threshold | 65% minimum to accept a prediction | Filters low-confidence noise while maintaining throughput |
| Class coverage | 29 classes (A-Z excluding motion-required J, Z; plus del, nothing, space) | Covers static alphabet signs |
| Model options | 4 interchangeable backbones: MobileNetV2, ResNet50, EfficientNet-B0, Custom CNN | Trade-off flexibility between speed and accuracy |
| Swapability | Model type selected via config; inference transforms match training transforms | Guarantees consistent behavior across model swaps |
| Prediction smoothing | Majority voting + stability counting + confidence gate | Layered defense against erratic predictions |

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
| Local mode data handling | Desktop inference is fully local; no external transmission | User privacy preserved for the desktop pipeline |
| Web mode data handling | Browser sends **cropped hand images** (not the full frame) to the API; nothing is persisted server-side | Minimises what leaves the client; images are processed in-memory only |
| Session isolation | Each WebSocket connection gets its own `SentenceBuilder` + `TemporalSmoother`, evicted on disconnect, with an LRU cap (`MAX_SESSIONS`) | Concurrent users never share state; bounds memory against session-id flooding |
| Input hardening | `MAX_IMAGE_BYTES` / `MAX_IMAGE_PIXELS` limits + PIL decompression-bomb guard on public endpoints | Rejects oversized/malicious payloads (DoS) |
| CORS | `allow_credentials` disabled under wildcard origins; `ALLOWED_ORIGINS` locks it to known domains | Prevents credentialed cross-origin abuse |
| Web deployment auth | Planned — no authentication yet | Prevents unauthorized model access when deployed |
| Model protection | `.onnx` served by the API; `.pth`/`.onnx` in `outputs/models/` (Git LFS) | Weights remain under repo owner control |

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
| Training pipeline | `crop_dataset.py` + `train_and_export.py` | Hand-crop preprocessing + training/ONNX export (notebook for exploration) |
| Inference runtime (web) | ONNX Runtime | Optimised CPU inference in the API |
| Target deployment | Live web application | End-user delivery channel |

## 4. Data Architecture

### Dataset Organization

```
datasets/
  asl_alphabet_train/     ← Primary training data
  asl_alphabet_test/      ← Primary test data
  asl_test_organized/     ← Additional organized test data
  combined_training/      ← Merged training dataset (~600 images/class × 29)
  combined_cropped/       ← MediaPipe hand crops of combined_training (ADR-007)
  custom_dataset/         ← User-captured images
```

### Data Flow

1. **Collection**: Webcam captures ROI-cropped images per class (~600 images/class, 29 classes ≈ 17.4k images)
2. **Hand-cropping**: `crop_dataset.py` re-crops the dataset to the MediaPipe hand region, matching the live app (ADR-007)
3. **Augmentation**: Random crop, horizontal flip, color jitter, rotation, and normalization during training
4. **Training**: Deterministic stratified per-class train/val split (no leakage across classes)
5. **Validation**: Held-out per-class split for accuracy metrics
6. **Inference**: Live frames processed with the same crop + transforms as training

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
→ Resize to 224×224 → Tensor → Normalize (ImageNet) → Model Forward Pass
→ Softmax Probabilities → Confidence Gate → Temporal Smoothing → Letter Output
```

### Temporal Smoothing

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Smoothing window | 5 frames | Majority voting over recent predictions |
| Stability frames | 12 frames | Consecutive same prediction to commit letter |
| Confidence threshold | 0.65 | Minimum probability for a prediction to count toward a commit |

> `COOLDOWN_FRAMES` remains in config for compatibility but is unused; the old frame-count cooldown was dead code and has been removed.

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
- API authentication for public deployments (multi-user session isolation is already implemented)
- Mobile deployment
- Model retraining pipeline automation
