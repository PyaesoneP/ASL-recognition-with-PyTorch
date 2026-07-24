# ASL Recognition System

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%20optional-green.svg)

Real-time American Sign Language (ASL) recognition using PyTorch and MediaPipe. Captures hand gestures via webcam, classifies them with deep learning models, and builds recognized sentences.

## Pipeline

```
Webcam → Hand Detection (MediaPipe) → ROI Crop → CNN Inference → Temporal Smoothing → Text Output
```

## Quick Start

### Local Inference (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference (MediaPipe hand detection)
python asl_pytorch_inference.py

# Fallback mode (fixed ROI, no MediaPipe)
python asl_pytorch_inference.py --simple

# Capture training images
python src/scripts/capture_asl_images.py
```

### Web App (Docker)

```bash
# Generate model weights (replaces Git LFS pointers)
.venv/bin/python generate_models.py

# Start stack — nginx frontend at http://localhost proxying the API container
docker compose up -d

# Stop
docker compose down
```

> **Deployment note:** `docker-compose.yml` + `frontend/nginx.conf` run a
> two-container setup (nginx serves the frontend and proxies `/api/` to the
> `api` service) for local use. Render instead deploys a **single** container
> (`render.yaml`) where FastAPI serves the frontend via `StaticFiles` — nginx
> is not used there.

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| MobileNetV2 | ~3.5M | Default — best speed/accuracy trade-off |
| ResNet50 | ~25M | Higher accuracy, higher latency |
| EfficientNet-B0 | ~5M | Lightweight alternative |
| Custom CNN | ~1M | Minimal footprint |

Switch models by updating `MODEL_TYPE` in `src/config/settings.py` or via env var:

```bash
MODEL_TYPE=resnet50 docker compose up -d
```

## Configuration

All tunable parameters live in `src/config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.65 | Minimum prediction confidence |
| `STABILITY_FRAMES` | 12 | Frames to hold before committing a letter |
| `COOLDOWN_FRAMES` | 18 | Legacy; letter repeats are now gated by a require-release debounce, not a frame count |
| `SMOOTHING_WINDOW` | 5 | Majority voting window size |
| `IMG_SIZE` | 224 | Input image size |

## Controls

**Desktop viewer** (`src/inference`, OpenCV window):

| Key | Action |
|-----|--------|
| Q | Quit |
| C | Clear sentence |
| SPACE | Add space |
| BACKSPACE | Delete last character |
| S | Save screenshot |
| R | Reset prediction history |

**Web app** (browser): `C` clear · `Space` add space · `Backspace` delete last.

## Web App Architecture

The web deployment splits the pipeline between browser and server:

| Client (Browser) | Server (API) |
|-----------------|--------------|
| Camera capture (`getUserMedia`) | Image preprocessing (`ImagePreprocessor`) |
| Hand detection (MediaPipe Hands JS) | CNN inference (ONNX Runtime) |
| ROI crop (canvas) | Temporal smoothing (`TemporalSmoother`) |
| DOM rendering | Sentence accumulation (`SentenceBuilder`) |
| | WebSocket streaming |

Single-container deployment: the API serves the frontend via FastAPI `StaticFiles` (no separate nginx needed for Render).
Multi-container deployment: Docker Compose with separate `api` (FastAPI) and `frontend` (nginx) services.

API endpoints: `/api/predict`, `/api/update`, `/api/sentence/*`, `/api/stream` (WebSocket).
See [`docs/AGENTS.md`](docs/AGENTS.md) for full endpoint reference.

## Project Structure

```
├── src/
│   ├── config/          # Centralized settings (env var overrides)
│   ├── inference/       # Local inference pipeline
│   ├── scripts/         # Capture and utility scripts
├── api/                 # Web API service (FastAPI)
│   ├── main.py          # FastAPI endpoints + WebSocket
│   ├── models.py        # Pydantic schemas
│   └── services/        # ImagePreprocessor, ASLPredictor, ModelRegistry, etc.
├── frontend/            # Browser client (MediaPipe JS + vanilla JS)
│   ├── index.html       # Main page
│   ├── css/             # Dark theme styling
│   └── js/              # MediaPipe integration, sentence manager
├── notebooks/           # Training notebooks
├── datasets/            # Training data
├── outputs/
│   ├── models/          # Trained model checkpoints
│   └── metrics/         # Training metrics
├── docs/                # Architecture documentation
├── generate_models.py   # Generate model weights from LFS pointers
├── docker-compose.yml   # Service orchestration
├── test_webapp.py       # 115 unit/integration tests
└── test_edge_cases.py   # 60 edge case tests
```

## Architecture

See [`docs/`](docs/) for detailed documentation:

- [Architectural Characteristics](docs/architectural_characteristics.md) — NFRs, performance targets, tech stack
- [Logical Components](docs/logical_components.md) — Component inventory, interfaces, dependency map
- [Architectural Decisions](docs/architectural_decisions.md) — 6 ADRs covering framework, training, detection, and deployment choices
- [Architectural Style](docs/architectural_style.md) — Pipeline design, stages, anti-patterns

## Requirements

- **OS**: Windows (primary), Linux with `--simple` fallback
- **Python**: 3.8+
- **GPU**: Optional (CUDA auto-detected; CPU fallback supported)

## Future Work

- Two-handed sign recognition (J, Z, numbers)
- Continuous gesture sequences beyond single letters
- Mobile deployment (TFLite/CoreML)
