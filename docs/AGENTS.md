# ASL Recognition System — Agent Reference

> **Rule: Always respect the architectural docs. All code changes must align with the pipeline stages, component boundaries, and ADRs defined below.**

## Architecture Docs (read before coding)

| Doc | Contents |
|-----|----------|
| [`architectural_style.md`](./architectural_style.md) | Pipeline stages 1–8, client/server split, anti-patterns, data flow |
| [`logical_components.md`](./logical_components.md) | 14 components, interfaces, dependency map |
| [`architectural_decisions.md`](./architectural_decisions.md) | 6 ADRs (PyTorch, transfer learning, MediaPipe, image-based, config-driven, local-first) |
| [`architectural_characteristics.md`](./architectural_characteristics.md) | NFRs, perf targets, tech stack, deployment |

## Project Structure

```
src/config/settings.py          ← Centralized config (ADR-005)
src/inference/__init__.py       ← Local inference pipeline (HandDetector, UIRenderer)
src/scripts/capture_asl_images.py ← Dataset capture
api/                            ← Web API service
  main.py                       ← FastAPI endpoints
  models.py                     ← Pydantic schemas
  services/predictor.py         ← ImagePreprocessor, ASLPredictor, TemporalSmoother, ModelRegistry, SentenceBuilder
  requirements.txt
  Dockerfile
frontend/                       ← Browser client (MediaPipe JS + vanilla JS)
notebooks/ASL_PyTorch_Complete.ipynb ← Training
outputs/models/                 ← Trained checkpoints (.pth) and ONNX exports (.onnx)
outputs/metrics/                ← Training metrics
```

## Pipeline Mapping (client ↔ server)

| Stage | Client (Browser) | Server (API) |
|-------|-----------------|--------------|
| 1 Capture | `getUserMedia()` | — |
| 2 Hand Detection | MediaPipe Hands JS | — |
| 3 ROI Extraction | Canvas crop | — |
| 4 Preprocessing | — | `ImagePreprocessor` |
| 5 Classification | — | `ASLPredictor` (Stage 5) |
| 6 Temporal Smoothing | — | `TemporalSmoother` |
| 7 Output Formatting | — | `SentenceBuilder` |
| 8 Rendering | DOM/CSS overlay | — |

## API Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/health` | — | `{status, model_type, model_loaded}` |
| POST | `/api/predict` | `{image, session_id}` | `{prediction, confidence, probabilities}` |
| POST | `/api/update` | `{class_label, confidence, session_id}` | `{sentence, added_letter, session_id}` |
| POST | `/api/sentence/update` | `{session_id, action, prediction}` | `{sentence, session_id}` |
| GET | `/api/sentence/{id}` | — | `{sentence, session_id}` |
| WS | `/api/stream` | JSON messages | JSON events |

## Key Constraints

- **Config-driven**: All thresholds/paths from `src/config/settings.py` — no magic numbers in code
- **29 classes**: A–Z + `del`, `nothing`, `space`
- **Smoothing params**: 5-frame window, 12 stability frames, 18 cooldown, 0.65 confidence
- **Input size**: 224×224, ImageNet mean/std normalization
- **4 model backbones**: MobileNetV2, ResNet50, EfficientNet-B0, CustomCNN (strategy via `ModelRegistry`)
- **Self-contained API**: `api/services/predictor.py` does not import from `src.inference` (avoids mediapipe segfault in server)
