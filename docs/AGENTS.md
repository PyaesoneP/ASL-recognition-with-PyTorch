# ASL Recognition System — Agent Reference

## Architecture Documentation

| Document | Purpose |
|----------|---------|
| [`architectural_characteristics.md`](./architectural_characteristics.md) | NFRs, performance targets, tech stack, data architecture, deployment |
| [`logical_components.md`](./logical_components.md) | Component inventory, interfaces, data flow, dependency map |
| [`architectural_decisions.md`](./architectural_decisions.md) | 6 ADRs: framework, training strategy, detection, classification, config, deployment |
| [`architectural_style.md`](./architectural_style.md) | Pipeline style, stage definitions, anti-patterns, web deployment mapping |

## Quick Start

```bash
pip install torch torchvision opencv-python mediapipe numpy

python asl_pytorch_inference.py          # MediaPipe mode
python asl_pytorch_inference.py --simple # Fallback mode
python src/scripts/capture_asl_images.py # Dataset capture
```

## Key Files

| Path | Role |
|------|------|
| `src/config/settings.py` | Centralized configuration |
| `src/inference/__init__.py` | Inference pipeline (HandDetector, ASLPredictor, SentenceBuilder, UIRenderer) |
| `src/scripts/capture_asl_images.py` | Image capture tool |
| `notebooks/ASL_PyTorch_Complete.ipynb` | Training notebook |
| `outputs/models/` | Trained model checkpoints |
| `outputs/metrics/` | Training metrics |

## Notes

- Run on Windows directly (not WSL2) for webcam access
- All config in `src/config/settings.py` — no magic numbers in code
- Inference transforms must match training transforms exactly
