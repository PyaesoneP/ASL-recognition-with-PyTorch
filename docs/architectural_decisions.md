# Architectural Decision Records — ASL Recognition System

---

## ADR-001: PyTorch Over TensorFlow/JAX

**Status**: Accepted
**Date**: 2026-01
**Context**: Selection of deep learning framework for model training and inference.

### Context

The project requires a deep learning framework capable of:
- Defining custom CNN architectures from scratch
- Loading and fine-tuning pretrained ImageNet models
- Running real-time inference at 30 FPS on consumer hardware
- Integrating with OpenCV for image preprocessing
- Supporting both CPU and GPU execution

Candidate frameworks evaluated: PyTorch, TensorFlow/Keras, JAX.

### Decision

Use **PyTorch** as the primary deep learning framework.

### Consequences

**Positive**:
- Imperative, Pythonic API aligns with the project's research-and-iterate development style
- `torchvision.models` provides direct access to MobileNetV2, ResNet50, and EfficientNet-B0 with pretrained weights
- Dynamic computation graph simplifies custom CNN definition (CustomCNN class)
- Seamless NumPy interoperability for OpenCV pipeline integration
- `torch.no_grad()` context enables efficient inference without graph overhead
- Jupyter Notebook integration is first-class, supporting the iterative training workflow

**Negative**:
- TensorFlow Serving and TFLite ecosystem not directly available for production deployment
- ONNX export required for cross-platform inference optimization
- Larger community around TensorFlow for enterprise MLOps tooling

**Status update**: ONNX export is now implemented — the web API serves the model via ONNX Runtime (see ADR-006), and `train_and_export.py` exports `.onnx` at the end of training.

---

## ADR-002: Transfer Learning Over Training From Scratch

**Status**: Accepted
**Date**: 2026-01
**Context**: Strategy for achieving accurate ASL classification with limited domain-specific training data.

### Context

At the time of this decision the ASL dataset contained ~100 images per class (~2,900 total); it has since grown to ~600 images/class (~17.4k total, `combined_training`). Even so, training deep CNNs from scratch typically requires far more data to converge, and the dataset comes from a single capture environment — so transfer learning remains the better fit for accuracy and generalization.

Options considered:
1. Train all models from scratch with random initialization
2. Use transfer learning from ImageNet pretrained weights
3. Hybrid: pretrained backbone + trained-from-scratch classifier

### Decision

Use **transfer learning from ImageNet pretrained weights** for MobileNetV2, ResNet50, and EfficientNet-B0. Retain a **custom CNN trained from scratch** as a lightweight baseline for comparison and minimal-footprint scenarios.

### Consequences

**Positive**:
- ImageNet pretrained features generalize well to hand gesture images (shared low-level features: edges, textures, shapes)
- Significantly fewer training epochs needed for convergence
- Better accuracy with the same dataset size
- Custom CNN from scratch provides a controlled baseline to measure transfer learning benefit

**Negative**:
- Domain gap between natural images (ImageNet) and hand gesture images may limit initial feature relevance
- Fine-tuning requires careful learning rate selection to avoid destroying pretrained features
- Pretrained models increase initial download size

**Mitigation**: Learning rate scheduling and selective layer freezing used during training; CustomCNN provides scratch-trained alternative.

---

## ADR-003: MediaPipe for Hand Detection

**Status**: Accepted
**Date**: 2026-01
**Context**: Selection of hand detection method for automatic ROI extraction from video frames.

### Context

The inference pipeline must detect hands in real-time video frames to crop the relevant region for classification. Requirements:
- Real-time performance at 30 FPS
- Robust detection across lighting conditions and hand poses
- Minimal integration complexity
- Available on Windows without specialized hardware

Options considered:
1. MediaPipe Hands (ML-powered landmark detector)
2. OpenCV Haar cascades / DNN module
3. Custom YOLO-based hand detector
4. Fixed ROI without detection (manual positioning)

### Decision

Use **MediaPipe Hands** as the primary hand detection method, with a **fixed-ROI fallback** (`--simple` mode) when MediaPipe is unavailable.

### Consequences

**Positive**:
- MediaPipe provides 21 detailed hand landmarks with sub-millisecond latency on CPU
- Built-in tracking reduces detection frequency, improving throughput
- Landmark-based bounding box computation allows precise hand cropping with configurable padding
- Lightweight dependency with no GPU requirement
- Graceful fallback to fixed-ROI mode ensures system operability without MediaPipe

**Negative**:
- MediaPipe adds an external dependency that may have version compatibility issues
- Detection can fail with occluded hands or extreme lighting
- Single-hand mode limits recognition to static single-hand signs

**Mitigation**: `--simple` flag bypasses MediaPipe entirely; configurable detection/tracking confidence thresholds allow tuning for different environments.

---

## ADR-004: Image-Based Classification Over Video-Based (3D CNN / LSTM)

**Status**: Accepted
**Date**: 2026-01
**Context**: Choice between frame-by-frame image classification and temporal video classification for ASL recognition.

### Context

ASL includes both static signs (most alphabet letters) and dynamic signs (J, Z, numbers requiring motion). The architectural choice affects model complexity, inference latency, and dataset requirements.

Options considered:
1. Per-frame 2D CNN classification with temporal post-processing
2. 3D CNN (e.g., C3D, I3D) operating on video clips
3. 2D CNN + LSTM/Transformer for sequence modeling
4. Two-stage: detect motion → route to static or dynamic classifier

### Decision

Use **per-frame 2D CNN classification** with **temporal smoothing as a post-processing step**. Dynamic signs (J, Z) are excluded from the current class set.

### Consequences

**Positive**:
- Simpler model architecture with well-understood training procedures
- Lower inference latency (single forward pass per frame vs. clip-based processing)
- Smaller memory footprint (no sequence buffering in model)
- Easier to swap model backbones (any 2D image classifier works)
- Temporal smoothing (majority voting, stability counting, confidence gate) provides sufficient stability for static signs
- Dataset requirements are simpler: individual images per class, not video sequences

**Negative**:
- Cannot recognize motion-based signs (J, Z, numbers) without architectural extension
- Temporal smoothing is heuristic-based, not learned
- No explicit modeling of gesture dynamics

**Mitigation**: Architecture is extensible — future two-stage design can add a motion detector that routes dynamic gestures to a sequence model while keeping the 2D CNN for static signs.

---

## ADR-005: Config-Driven Model Selection

**Status**: Accepted
**Date**: 2026-01
**Context**: Mechanism for selecting and swapping model architectures without code modification.

### Context

The system supports four model backbones (MobileNetV2, ResNet50, EfficientNet-B0, Custom CNN). Users and developers need to experiment with different models to evaluate speed/accuracy trade-offs. Hardcoding model selection in the inference pipeline would require code changes and testing for each swap.

Options considered:
1. Hardcode model type in inference code
2. Command-line argument for model selection
3. Centralized configuration file with model type and path
4. Model registry with auto-discovery

### Decision

Use a **centralized configuration module** (`src/config/settings.py`) that defines `MODEL_TYPES`, `DEFAULT_MODEL_PATH`, and `PREDICTION_DEFAULTS`. The inference layer reads from config but allows per-run overrides at the top of `src/inference/__init__.py`.

### Consequences

**Positive**:
- Model swap requires editing a single config file, not inference code
- Prediction parameters (confidence threshold, stability frames, cooldown, smoothing window) are co-located with model settings
- Dataset paths are similarly centralized, enabling dataset experimentation
- Two-level override: config defaults → inference-level overrides → future CLI arguments
- Reduces risk of train/inference mismatch (transforms and model type referenced from same config)

**Negative**:
- Config values are Python constants, not externalized to JSON/YAML (would require additional parsing)
- Changes require restart of inference process (no hot-reload)

**Mitigation**: Future enhancement can externalize config to YAML/JSON while preserving the centralized access pattern.

---

## ADR-006: Local-First Inference Design

**Status**: Accepted
**Date**: 2026-01
**Context**: Deployment model for the inference pipeline — local execution vs. cloud-hosted service.

### Context

The system processes live webcam video containing personal visual data. The deployment model affects latency, privacy, cost, and accessibility.

Options considered:
1. Fully cloud-hosted: stream video to server, return predictions
2. Fully local: all processing on user's device
3. Local-first with optional cloud features

### Decision

Design for **local-first inference** with all processing (hand detection, image preprocessing, model inference, temporal smoothing) running on the user's machine. Plan for a **web deployment path** that preserves local capture and moves only model inference to a server.

### Consequences

**Positive**:
- Zero latency from capture to display (no network round-trip)
- User's webcam footage never leaves the local machine
- No cloud hosting costs during development and personal use
- Works offline without internet connectivity
- Model weights are small enough (~14MB for MobileNetV2) to bundle with application

**Negative**:
- Inference speed depends on user's hardware (CPU-only users will be slower)
- Model updates require manual download and replacement
- No centralized analytics or usage telemetry
- Web deployment requires architectural refactoring to separate capture from inference

**Mitigation**: Web deployment path is implemented. The API service (`api/services/predictor.py`) provides ONNX Runtime inference with session-based sentence building. Browser client handles stages 1-3 (capture, detection, ROI) via MediaPipe JS. Single-container deployment on Render via FastAPI + StaticFiles.

---

## ADR-007: MediaPipe Hand-Crop Training for Train/Serve Parity

**Status**: Accepted
**Date**: 2026-07
**Context**: Closing the gap between how images are framed during training and at inference.

### Context

The browser feeds the model a **tight MediaPipe hand crop** (square bounding box around the 21 landmarks, 25% padding), but the raw `combined_training` images are full webcam frames with background and forearm. A model trained on full frames reached high in-distribution validation accuracy yet generalized poorly to the live app — a classic train/serve distribution mismatch, compounded by the dataset coming from a single capture environment.

Options considered:
1. Train on full frames, crop only at inference (status quo — mismatch)
2. Send full frames to the model at inference (background/lighting dependence)
3. Crop the dataset the **same way** the app crops live frames, then train on crops

### Decision

Preprocess the dataset through MediaPipe into hand crops (`crop_dataset.py`) using the **exact bbox math as the frontend** (`getBounds`/`cropAndPredict`), and train on `datasets/combined_cropped`. Images with no detected hand fall back to a full-frame resize so no class loses samples.

### Consequences

**Positive**:
- Training input matches inference input — the model sees the same framing both sides
- Removes most background/scene dependence, improving live-webcam generalization
- Deterministic and reproducible (`crop_dataset.py` is a one-shot batch step)

**Negative**:
- Adds a preprocessing step and a MediaPipe Tasks model dependency (`hand_landmarker.task`)
- Classes with low hand-detection rates (e.g. `nothing`) rely on the full-frame fallback
- In-distribution validation accuracy (≈99.8%) still overstates real-world accuracy given the single-environment dataset

---

## ADR-008: Per-Session Server State

**Status**: Accepted
**Date**: 2026-07
**Context**: Isolating prediction/sentence state between concurrent web clients.

### Context

The initial web API used one global `TemporalSmoother` and a `"default"` sentence session shared by every WebSocket connection, so concurrent users cross-contaminated each other's smoothing and sentence. Client-supplied `session_id`s also created unbounded server state.

### Decision

Give **each WebSocket connection a unique session** (uuid) with its own `SentenceBuilder` + `TemporalSmoother`, evicted on disconnect. Store sessions in an **LRU-capped** map (`MAX_SESSIONS`) so client-supplied ids can't exhaust memory.

### Consequences

**Positive**:
- Concurrent users are fully isolated; smoothing and sentences never mix
- Bounded memory under session-id flooding; automatic cleanup on disconnect

**Negative**:
- Sessions are in-process only — horizontal scaling would need shared/sticky state
- The HTTP endpoints still accept arbitrary `session_id`s (bounded, but not authenticated)

---

## ADR Index

| ID | Topic | Status |
|----|-------|--------|
| ADR-001 | PyTorch over TensorFlow/JAX | Accepted |
| ADR-002 | Transfer learning over training from scratch | Accepted |
| ADR-003 | MediaPipe for hand detection | Accepted |
| ADR-004 | Image-based classification over video-based | Accepted |
| ADR-005 | Config-driven model selection | Accepted |
| ADR-006 | Local-first inference design | Accepted |
| ADR-007 | MediaPipe hand-crop train/serve parity | Accepted |
| ADR-008 | Per-session server state | Accepted |
