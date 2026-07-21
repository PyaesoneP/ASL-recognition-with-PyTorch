# ASL Recognition with PyTorch - Development Guide

## Quick Start

```bash
# Install dependencies
pip install torch torchvision opencv-python mediapipe numpy

# Run inference app (MediaPipe)
python asl_pytorch_inference.py

# Run simple mode (no MediaPipe, manual ROI)
python asl_pytorch_inference.py --simple

# Capture training images
python capture_asl_images.py
```

## Critical Configuration

Update these in `asl_pytorch_inference.py` before running:

- **MODEL_PATH**: Path to your `.pth` model file
- **MODEL_TYPE**: One of `mobilenet_v2`, `resnet50`, `efficientnet_b0`, `custom_cnn`
- **IMG_SIZE**: Must match training (default: 224)
- **CLASS_NAMES**: Order must exactly match training class order

## Training Data Structure

The dataset is organized as:
```
<dataset_name>/
  <class_name>/
    img_001.jpg
    img_002.jpg
    ...
```

Current datasets:
- `asl_alphabet_train` / `asl_alphabet_test` - Alphabet training split
- `asl_test_organized` / `combined_training` - Additional data
- `custom_dataset` - Custom captures (created by tool)

## Model Types & Requirements

| Type | Description | Training Required? |
|------|-------------|-------------------|
| `mobilenet_v2` | PyTorch MobileNetV2 with custom classifier | Yes |
| `resnet50` | PyTorch ResNet50 with custom fc layer | Yes |
| `efficientnet_b0` | EfficientNet-B0 with custom head | Yes |
| `custom_cnn` | Custom CNN (3 blocks, 256-class output) | Yes |

**IMPORTANT**: Inference transforms must match training transforms exactly.

## Device Requirement

This application **must run on Windows**, NOT in WSL2. If using WSL2, use:
- `main_simple()` function for basic operation
- Or run directly from Windows host

## Key Prediction Settings

Adjust these for different gesture recognition needs:

- `CONFIDENCE_THRESHOLD`: 0.65 (minimum to accept prediction)
- `STABILITY_FRAMES`: 12 (~0.4s at 30fps) to confirm a letter
- `COOLDOWN_FRAMES`: 18 frames between adding letters
- `SMOOTHING_WINDOW`: 5 frames for majority voting

## Inference Flow

1. MediaPipe detects hand and crops ROI with 25% padding
2. Crops resized to square, centered
3. Image transformed (resize→tensor→normalize)
4. Model predicts class probabilities
5. Majority voting smoothing over `SMOOTHING_WINDOW` frames
6. Letter added after `STABILITY_FRAMES` consecutive same predictions

## Controls (Inference App)

- **Q**: Quit
- **C**: Clear sentence
- **SPACE**: Add space between words
- **BACKSPACE**: Delete last character
- **S**: Save screenshot
- **R**: Reset prediction history
