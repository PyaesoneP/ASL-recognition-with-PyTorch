# Logical Components — ASL Recognition System

## Component Inventory

The system is organized into 14 logical components grouped by responsibility domain.

---

## 1. Data Acquisition Layer

### 1.1 DataCaptureModule

**Location**: `src/scripts/capture_asl_images.py`

**Responsibility**: Acquires raw image data from webcam for dataset construction.

**Modes**:
- **Interactive capture**: User specifies class name, captures images via SPACE key, targets 100 images per class
- **Batch capture**: Auto-captures predefined set of classes at fixed rate (100ms intervals)

**Key behaviors**:
- Opens webcam at 640×480 resolution
- Draws 300×300 ROI rectangle at fixed position (x=100, y=50)
- Displays live preview of cropped region
- Saves ROI images with timestamp-based filenames
- Reports per-class and total summary on completion

**Interfaces**:
- Input: class name string, target count integer
- Output: JPEG files organized in `datasets/<class_name>/` directory structure

---

### 1.2 DatasetManager

**Location**: `src/config/settings.py` (DATASETS dict)

**Responsibility**: Central registry of dataset paths and their roles.

**Managed datasets**:

| Key | Path | Role |
|-----|------|------|
| `train` | `datasets/asl_alphabet_train/` | Primary training split |
| `test` | `datasets/asl_alphabet_test/` | Primary evaluation split |
| `organized` | `datasets/asl_test_organized/` | Additional test data |
| `combined` | `datasets/combined_training/` | Merged training set |
| `custom` | `datasets/custom_dataset/` | User-captured images |

**Interfaces**:
- Provides path lookup by dataset key
- Extensible: new entries added without code changes

---

### 1.3 DataAugmenter

**Location**: Training notebook (`notebooks/ASL_PyTorch_Complete.ipynb`)

**Responsibility**: Applies randomized transformations to training data to improve model generalization.

**Transformations**:
- Random horizontal flip
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation
- Resize to 224×224
- Normalization with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Interfaces**:
- Input: raw PIL image
- Output: augmented tensor ready for model input

---

### 1.4 DataLoader

**Location**: Training notebook

**Responsibility**: Batches and feeds training data to the model with class-balanced sampling.

**Key behaviors**:
- Uses weighted oversampling to handle class imbalance
- Constructs batches from combined dataset
- Maintains separate validation split

**Interfaces**:
- Input: dataset paths, batch size, sampling weights
- Output: tensor batches of shape (batch_size, 3, 224, 224)

---

## 2. Model Layer

### 2.1 ModelRegistry

**Location**: `src/inference/__init__.py` (load_model function + model classes)

**Responsibility**: Defines, instantiates, and loads trained model architectures.

**Registered models**:

| Model | Architecture | Source | Parameters |
|-------|-------------|--------|-----------|
| MobileNetV2 | `models.mobilenet_v2` | PyTorch pretrained | ~3.5M |
| ResNet50 | `models.resnet50` | PyTorch pretrained | ~25M |
| EfficientNet-B0 | `models.efficientnet_b0` | PyTorch pretrained | ~5M |
| Custom CNN | `CustomCNN` class | Defined in-code | ~1M |

**Custom CNN details**:
- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization and ReLU after each convolution
- Max pooling (2×2) and dropout (0.25) per block
- Global average pooling
- 3-layer classifier (256→512→256→29) with dropout (0.5)

**Model loading logic**:
- Selects architecture based on `MODEL_TYPE` config
- Replaces final classifier layer for 29-class output
- Loads `.pth` checkpoint (supports both `model_state_dict` key and raw state dict formats)
- Moves model to CUDA if available, sets to eval mode

**Interfaces**:
- Input: model path string, model type string, number of classes
- Output: initialized `nn.Module` on correct device

---

### 2.2 TrainingEngine

**Location**: Training notebook

**Responsibility**: Orchestrates model training loop.

**Key behaviors**:
- Iterates over data loader batches
- Computes loss (cross-entropy) and backpropagates
- Applies optimizer step (Adam or SGD)
- Tracks per-epoch accuracy and loss
- Saves best model checkpoint to `outputs/models/`

**Interfaces**:
- Input: model, dataloaders, optimizer, loss function, epochs
- Output: trained model weights, training metrics

---

### 2.3 ValidationEngine

**Location**: Training notebook

**Responsibility**: Evaluates model on held-out test data.

**Key behaviors**:
- Runs inference in `torch.no_grad()` mode
- Computes classification accuracy
- Generates confusion matrix
- Logs per-class precision/recall

**Interfaces**:
- Input: model, test dataloader, class names
- Output: accuracy scalar, confusion matrix, per-class metrics

---

### 2.4 MetricsExporter

**Location**: Training notebook

**Responsibility**: Persists training artifacts for analysis.

**Outputs**:
- Accuracy curves to `outputs/metrics/`
- Loss curves to `outputs/metrics/`
- Confusion matrices to `outputs/metrics/`
- Best model checkpoint to `outputs/models/`

---

## 3. Detection & Preprocessing Layer

### 3.1 HandDetector

**Location**: `src/inference/__init__.py` (HandDetector class)

**Responsibility**: Detects hands in video frames using MediaPipe and computes bounding boxes.

**Configuration**:
- `max_hands`: 1 (single-hand mode)
- `detection_confidence`: 0.7
- `tracking_confidence`: 0.5
- `static_image_mode`: False (video stream optimization)

**Key behaviors**:
- Converts BGR frame to RGB for MediaPipe
- Processes landmarks and computes bounding box from min/max coordinates
- Adds 25% padding around detected hand
- Converts bounding box to square by using larger dimension and centering
- Clips box to frame boundaries
- Optionally draws landmark connections on frame

**Interfaces**:
- Input: OpenCV BGR frame, draw boolean
- Output: annotated frame, list of `(x, y, w, h)` bounding boxes

---

### 3.2 ROICropper

**Location**: `src/inference/__init__.py` (inline in `main()` loop, lines 633-637)

**Responsibility**: Extracts hand region from frame using bounding box coordinates.

**Key behaviors**:
- Uses first detected hand box
- Extracts sub-image: `frame[y:y+bh, x:x+bw]`
- Validates extracted region has sufficient size (>10×10 pixels)

**Simple mode variant**: Fixed ROI at (x=50, y=100) with 300×300 size

**Interfaces**:
- Input: frame, bounding box tuple
- Output: cropped hand image (BGR numpy array)

---

### 3.3 ImagePreprocessor

**Location**: `src/inference/__init__.py` (ASLPredictor.preprocess method + inference_transforms)

**Responsibility**: Transforms raw cropped image into model-ready tensor.

**Transform pipeline**:
1. BGR → RGB conversion (OpenCV)
2. ToPILImage
3. Resize to 224×224
4. ToTensor (values scaled to [0, 1])
5. Normalize with ImageNet mean/std
6. Add batch dimension
7. Move to target device (CPU/CUDA)

**Interfaces**:
- Input: BGR numpy array (cropped hand region)
- Output: tensor of shape (1, 3, 224, 224) on correct device

---

## 4. Inference Layer

### 4.1 InferenceEngine

**Location**: `src/inference/__init__.py` (ASLPredictor class)

**Responsibility**: Runs model forward pass and returns prediction with confidence.

**Key behaviors**:
- Wraps inference in `torch.no_grad()` for efficiency
- Applies softmax to raw logits
- Extracts top prediction class and confidence
- Maintains prediction history deque (maxlen=SMOOTHING_WINDOW)
- Applies majority voting when history has ≥3 entries

**Interfaces**:
- Input: BGR numpy array (hand crop)
- Output: `(predicted_class_str, confidence_float, probabilities_array)`

---

### 4.2 TemporalSmoother

**Location**: Embedded in ASLPredictor and SentenceBuilder

**Responsibility**: Reduces prediction instability through multi-layer smoothing.

**Three smoothing layers**:

| Layer | Mechanism | Window | Effect |
|-------|-----------|--------|--------|
| Majority voting | Counter.most_common on history deque | 5 frames | Reduces frame-to-frame flicker |
| Stability counting | Consecutive same-prediction counter | 12 frames | Confirms sustained gesture |
| Cooldown | Post-commit lockout | 18 frames | Prevents double-counting |

**Interfaces**:
- Input: stream of (class, confidence) tuples
- Output: filtered letter commits

---

### 4.3 OutputFormatter

**Location**: `src/inference/__init__.py` (SentenceBuilder class)

**Responsibility**: Accumulates recognized letters into coherent sentences.

**Key behaviors**:
- Handles special classes: `space` → space character, `del` → backspace
- Filters predictions below confidence threshold
- Ignores `nothing` class (no hand present)
- Tracks current letter, stability count, and cooldown state
- Provides progress indicator (0.0 to 1.0) toward next letter commit

**Manual controls**:
- `add_space()`: inserts space character
- `delete_last()`: removes last character
- `clear()`: resets entire sentence and state

**Interfaces**:
- Input: predicted letter string, confidence float
- Output: `(added_letter_or_None, current_sentence_string)`

---

## 5. Presentation Layer

### 5.1 UIRenderer

**Location**: `src/inference/__init__.py` (UIRenderer class)

**Responsibility**: Draws all visual overlays on video frames.

**Rendered elements**:

| Method | Element | Details |
|--------|---------|---------|
| `draw_hand_box` | Bounding box | Green rectangle around detected hand |
| `draw_prediction` | Prediction label | Letter + confidence %, color-coded by confidence |
| `draw_sentence_box` | Sentence display | Dark box at frame bottom with accumulated text |
| `draw_progress_bar` | Stability progress | Horizontal bar showing letter confirmation progress |
| `draw_top_predictions` | Top-5 bar chart | Horizontal bars with class names and percentages |
| `draw_instructions` | Control hints | Keyboard shortcut reference at top-left |
| `draw_status` | Status message | "No hand detected" or other status text |
| `draw_feedback` | Letter flash | Large "+ LETTER" overlay on commit |

**Color coding**:
- Green (0, 255, 0): confidence ≥ 0.8
- Yellow (0, 255, 255): confidence 0.6–0.8
- Orange (0, 165, 255): confidence < 0.6

---

## 6. Configuration Layer

### 6.1 ConfigurationManager

**Location**: `src/config/settings.py`

**Responsibility**: Centralized configuration for all system parameters.

**Managed settings**:

| Category | Parameters |
|----------|-----------|
| Paths | `BASE_DIR`, `DEFAULT_MODEL_PATH` |
| Model selection | `MODEL_TYPES` list |
| Dataset registry | `DATASETS` dict (5 entries) |
| Prediction tuning | `CONFIDENCE_THRESHOLD`, `STABILITY_FRAMES`, `COOLDOWN_FRAMES`, `SMOOTHING_WINDOW`, `IMG_SIZE` |

**Inference-layer overrides**: `src/inference/__init__.py` allows per-run overrides of `MODEL_PATH`, `MODEL_TYPE`, and prediction parameters.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                             │
│                                                                     │
│  DataCaptureModule ────→ datasets/<class>/img_*.jpg                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING (notebook)                           │
│                                                                     │
│  DataLoader ──→ DataAugmenter ──→ TrainingEngine ──→ Model          │
│       │                                                    │        │
│       └──── ValidationEngine ──→ MetricsExporter           │        │
│                                                           ▼        │
│                                                    outputs/models/ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                               │
│                                                                     │
│  Webcam → HandDetector → ROICropper → ImagePreprocessor             │
│                                    │              │                 │
│                              ModelRegistry ← ConfigurationManager  │
│                                    │              │                 │
│                              InferenceEngine → TemporalSmoother     │
│                                               │                     │
│                                       OutputFormatter               │
│                                               │                     │
│                                       UIRenderer → Display          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Dependency Map

```
DataCaptureModule
  └─ depends on: OpenCV (webcam), OS filesystem

DatasetManager
  └─ depends on: filesystem paths (datasets/)

DataAugmenter
  └─ depends on: torchvision.transforms

DataLoader
  └─ depends on: DatasetManager, DataAugmenter, PyTorch DataLoader

ModelRegistry
  └─ depends on: PyTorch models, ConfigurationManager (MODEL_TYPE, MODEL_PATH)

TrainingEngine
  └─ depends on: ModelRegistry, DataLoader, ValidationEngine

ValidationEngine
  └─ depends on: ModelRegistry, DataLoader, MetricsExporter

MetricsExporter
  └─ depends on: filesystem (outputs/metrics/)

HandDetector
  └─ depends on: MediaPipe, OpenCV

ROICropper
  └─ depends on: HandDetector (bounding box output)

ImagePreprocessor
  └─ depends on: torchvision.transforms, ConfigurationManager (IMG_SIZE)

InferenceEngine
  └─ depends on: ModelRegistry, ImagePreprocessor, ConfigurationManager

TemporalSmoother
  └─ depends on: InferenceEngine output, ConfigurationManager (thresholds)

OutputFormatter (SentenceBuilder)
  └─ depends on: TemporalSmoother output, ConfigurationManager

UIRenderer
  └─ depends on: HandDetector, InferenceEngine, OutputFormatter, OpenCV

ConfigurationManager
  └─ depends on: none (leaf dependency)
```

## Component Interfaces Summary

| Component | Input | Output |
|-----------|-------|--------|
| DataCaptureModule | class name, target count | JPEG files in dataset directory |
| DatasetManager | dataset key | filesystem path string |
| DataAugmenter | PIL image | augmented tensor |
| DataLoader | dataset path, batch size | tensor batches |
| ModelRegistry | model path, type, num_classes | nn.Module instance |
| TrainingEngine | model, dataloaders, optimizer | trained weights, metrics |
| ValidationEngine | model, test loader, class names | accuracy, confusion matrix |
| MetricsExporter | metrics data | files in outputs/metrics/ |
| HandDetector | BGR frame | annotated frame, bounding boxes |
| ROICropper | frame, bounding box | cropped BGR array |
| ImagePreprocessor | BGR array | batched tensor on device |
| InferenceEngine | BGR array | (class, confidence, probabilities) |
| TemporalSmoother | prediction stream | filtered letter commits |
| OutputFormatter | letter, confidence | (added_letter, sentence) |
| UIRenderer | frame, prediction data | annotated frame |
| ConfigurationManager | none | config values |
