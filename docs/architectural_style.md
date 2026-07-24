# Architectural Style — ASL Recognition System

## Primary Style: Pipeline / Stream Processing

The ASL recognition system follows a **Pipeline / Stream Processing** architectural style. Data flows unidirectionally through a sequence of specialized stages, each transforming the data toward the final output of recognized text.

### Why Pipeline?

The problem domain — real-time gesture recognition from a video stream — is inherently sequential and stage-oriented:

1. Raw pixel data arrives continuously from a webcam
2. Each frame must pass through detection, cropping, preprocessing, classification, and smoothing before producing output
3. Stages have well-defined input/output contracts
4. Throughput (frames per second) is the primary performance concern
5. Stages can be independently optimized or replaced

---

## Stage Definitions

### Stage 1: Capture

**Input**: None (webcam device)
**Output**: Raw BGR frame (numpy array, 640×480×3)
**Component**: OpenCV `VideoCapture`

Responsibility: Acquire frames at target frame rate (30 FPS). Applies horizontal flip for mirror-mode display.

### Stage 2: Hand Detection

**Input**: Raw BGR frame
**Output**: Annotated frame + list of bounding box tuples `(x, y, w, h)`
**Component**: `HandDetector` (MediaPipe)

Responsibility: Locate hand landmarks and compute bounding regions. Draws landmark connections on frame for visual feedback.

### Stage 3: ROI Extraction

**Input**: Frame + bounding box
**Output**: Cropped BGR sub-image (variable size)
**Component**: Inline in main loop (`ROICropper` logical component)

Responsibility: Extract the hand region of interest. Validates that crop has sufficient dimensions.

### Stage 4: Preprocessing

**Input**: Cropped BGR array
**Output**: Batched tensor `(1, 3, 224, 224)` on target device
**Component**: `ImagePreprocessor` / `ASLPredictor.preprocess()`

Responsibility: Color space conversion, resize, tensor conversion, normalization, device placement.

### Stage 5: Classification

**Input**: Preprocessed tensor batch
**Output**: Class probabilities (softmax), predicted class, confidence
**Component**: `InferenceEngine` / `ASLPredictor.predict()`

Responsibility: Model forward pass. Applies softmax to logits. Maintains prediction history for smoothing.

### Stage 6: Temporal Smoothing

**Input**: Stream of (class, confidence) tuples
**Output**: Filtered letter commits
**Component**: `TemporalSmoother` / `SentenceBuilder.update()`

Responsibility: Layered smoothing — majority voting (5-frame window), stability counting (12 frames), and a confidence gate (`CONFIDENCE_THRESHOLD`). (`COOLDOWN_FRAMES` is legacy/unused.)

### Stage 7: Output Formatting

**Input**: Filtered letter commits
**Output**: Accumulated sentence string
**Component**: `OutputFormatter` / `SentenceBuilder`

Responsibility: Builds sentence from committed letters. Handles special classes (space, del). Provides manual editing operations.

### Stage 8: Rendering

**Input**: Frame + prediction state + sentence state
**Output**: Annotated display frame
**Component**: `UIRenderer`

Responsibility: Draws all visual overlays — hand box, prediction label, progress bar, top-5 predictions, sentence box, controls hint.

---

## Data Flow

```
Frame ──→ HandDetector ──→ (frame, boxes)
                              │
                              ▼
                        ROICropper ──→ cropped_image
                              │
                              ▼
                    ImagePreprocessor ──→ tensor_batch
                              │
                              ▼
                      InferenceEngine ──→ (class, confidence, probs)
                              │
                              ▼
                    TemporalSmoother ──→ letter_commit (or None)
                              │
                              ▼
                      OutputFormatter ──→ sentence_string
                              │
                              ▼
                        UIRenderer ──→ display_frame ──→ Monitor
```

Each stage passes data downstream. No stage writes to a shared state except the final OutputFormatter, which maintains the accumulated sentence.

---

## Separation of Concerns

The pipeline style enforces clean boundaries:

| Stage | Knows About | Does Not Know About |
|-------|-----------|-------------------|
| HandDetector | MediaPipe, frame dimensions | Model architecture, class names |
| ROICropper | Bounding box coordinates | Model, classes, smoothing |
| ImagePreprocessor | Image transforms, device | Hand detection, sentence state |
| InferenceEngine | Model forward pass, softmax | Smoothing parameters, UI |
| TemporalSmoother | Confidence thresholds, frame counts | Model internals, rendering |
| OutputFormatter | Class names, special actions | Image processing, detection |
| UIRenderer | Frame drawing, colors | Model, detection, smoothing logic |

This separation enables:
- **Independent testing**: Each stage can be unit-tested with mock inputs
- **Stage replacement**: Swap MediaPipe for another detector without touching inference
- **Parallel development**: Teams can work on different stages simultaneously
- **Performance profiling**: Bottlenecks isolated to specific stages

---

## Web Deployment (Implemented)

The pipeline architecture maps to a client-server web deployment:

### Client-Side (Browser)

| Stage | Implementation |
|-------|---------------|
| Stage 1: Capture | Browser `getUserMedia()` / WebRTC |
| Stage 2: Hand Detection | MediaPipe Hands JS |
| Stage 3: ROI Extraction | Canvas crop in browser |
| Stage 8: Rendering | HTML/CSS overlay on video element |

### Server-Side (API)

| Stage | Implementation |
|-------|---------------|
| Stage 4: Preprocessing | `ImagePreprocessor` (ONNX-compatible transforms) |
| Stage 5: Classification | `OnnxInferenceEngine` + `ASLPredictor` (ONNX Runtime) |
| Stage 6: Temporal Smoothing | `TemporalSmoother` (server-side stateful) |
| Stage 7: Output Formatting | `SentenceBuilder` (session-based) |

### Communication

The browser detects and crops the hand (Stages 1–3) and streams the cropped
image to the server, which runs classification **and** sentence-building
(Stages 4–7) in one round trip and returns the updated transcript. The live
frontend uses the WebSocket path:

```
Browser ──{action:"predict", image}──▶ WS /api/stream
Browser ◀──{prediction, confidence, sentence}── (server classifies + builds sentence)
```

Equivalent request/response HTTP endpoints exist for non-streaming clients and
tests: `POST /api/predict` (classification only → `{class, confidence}`) and
`POST /api/update` (`{class, confidence}` → `{sentence, added_letter}`). All
paths share the same per-session `SentenceBuilder` + `TemporalSmoother`.

The pipeline's stage boundaries align with natural API boundaries. Stages 1-3 move to the client (reducing bandwidth), while Stages 4-7 remain server-side (protecting the model).

### Deployment Modes

| Mode | Architecture | Use Case |
|------|-------------|----------|
| Single-container | FastAPI + StaticFiles (Render) | Simple deployment, one service |
| Multi-container | Docker Compose (api + nginx) | Development, separate scaling |

---

## Anti-Patterns Avoided

### 1. God Object

**Anti-pattern**: A single class or function that handles capture, detection, inference, smoothing, and rendering.

**How avoided**: Each pipeline stage is a distinct class (`HandDetector`, `ASLPredictor`, `SentenceBuilder`, `UIRenderer`). The main loop orchestrates but does not implement stage logic.

### 2. Shared Mutable State

**Anti-pattern**: Stages reading and writing a global state object, creating hidden dependencies and race conditions.

**How avoided**: Data flows explicitly through function parameters and return values. Mutable state is confined to the output accumulators — `SentenceBuilder.sentence` and the per-frame `TemporalSmoother` history. In the web API these are **per-session** (keyed by connection), so concurrent users never share mutable state.

### 3. Tight Coupling Between Detection and Classification

**Anti-pattern**: The classifier depending on MediaPipe internals or the detector depending on model architecture.

**How avoided**: HandDetector outputs generic bounding boxes. InferenceEngine accepts any cropped image. The contract between stages is a numpy array — no framework-specific types leak across boundaries.

### 4. Hardcoded Parameters

**Anti-pattern**: Threshold values, model paths, and image sizes embedded in stage implementations.

**How avoided**: `ConfigurationManager` (`src/config/settings.py`) centralizes all tunable parameters. Stages read from config; no magic numbers in stage code.

### 5. Monolithic Training Script

**Anti-pattern**: Training, validation, and metrics export entangled in a single script with no separation.

**How avoided**: Logical separation into `TrainingEngine`, `ValidationEngine`, `MetricsExporter`, and `DataLoader`. Though currently in a notebook, the components are identifiable and extractable into modules for future refactoring.

### 6. Synchronous Blocking Pipeline

**Anti-pattern**: Each stage blocks waiting for the previous stage, with no buffering or parallelism.

**Current state**: The pipeline is synchronous (one frame at a time), which is acceptable for the current 30 FPS target on consumer hardware. The stage-based design allows future introduction of producer-consumer queues between stages without changing stage implementations.

---

## Secondary Styles

### Component Architecture (within stages)

Each pipeline stage is implemented as a reusable component with:
- Well-defined constructor (configuration)
- Public methods (stage operations)
- Private state (buffers, counters, model references)
- Release/cleanup methods (resource management)

### Configuration Object Pattern

`ConfigurationManager` serves as a centralized configuration object. All stages reference the same config values, ensuring consistency across the pipeline.

### Strategy Pattern (model selection)

`ModelRegistry.get_model()` implements a strategy pattern: the same loading interface works for any registered model type. The `MODEL_TYPE` config value selects which strategy (architecture) to instantiate. Registered models: `mobilenet_v2`, `resnet50`, `efficientnet_b0`, `custom_cnn`.
