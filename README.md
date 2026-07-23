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

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| MobileNetV2 | ~3.5M | Default — best speed/accuracy trade-off |
| ResNet50 | ~25M | Higher accuracy, higher latency |
| EfficientNet-B0 | ~5M | Lightweight alternative |
| Custom CNN | ~1M | Minimal footprint |

Switch models by updating `MODEL_TYPE` and `MODEL_PATH` in `src/inference/__init__.py`.

## Configuration

All tunable parameters live in `src/config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.65 | Minimum prediction confidence |
| `STABILITY_FRAMES` | 12 | Frames to hold before committing a letter |
| `COOLDOWN_FRAMES` | 18 | Frames between letter additions |
| `SMOOTHING_WINDOW` | 5 | Majority voting window size |
| `IMG_SIZE` | 224 | Input image size |

## Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| C | Clear sentence |
| SPACE | Add space |
| BACKSPACE | Delete last character |
| S | Save screenshot |
| R | Reset prediction history |

## Project Structure

```
├── src/
│   ├── config/          # Centralized settings
│   ├── inference/       # Inference pipeline
│   ├── scripts/         # Capture and utility scripts
│   └── data/            # Data utilities
├── notebooks/           # Training notebooks
├── datasets/            # Training data
├── outputs/
│   ├── models/          # Trained model checkpoints
│   └── metrics/         # Training metrics
├── docs/                # Architecture documentation
└── requirements.txt
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
- Web deployment via containerized API
- Mobile deployment
