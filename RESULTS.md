# Results

Reproducible results for the ASL recognition system. Every number here is traceable to
a committed artifact — [outputs/train.log](outputs/train.log), the training notebook's
executed output cells, or a `benchmark.py` run — not to prose. Where a number is not yet
measured, it says so explicitly rather than guessing.

> **Two distinct training runs are reported below — do not conflate them.**
> **Experiment A** is a 4-model *architecture comparison* trained on the **uncropped**
> Kaggle images (in the notebook). **Experiment B** is the **deployed** model, retrained
> on **MediaPipe hand-crops** for train/serve parity (this is what the app actually
> serves). The A numbers are *not* the deployed model's numbers.

---

## Dataset

| | |
|---|---|
| Source | Kaggle **ASL Alphabet** (~87K images) + **2,899** custom webcam captures |
| Classes | **29** — A–Z, `del`, `nothing`, `space` |
| Deployed training set | `datasets/combined_cropped` — **17,399** images, MediaPipe hand-cropped to 224×224 ([crop_dataset.py](crop_dataset.py)) |
| Split (deployed) | **80 / 10 / 10** train/val/**test**, stratified within each class, seed 42 ([train_and_export.py](train_and_export.py)) → 13,921 / 1,739 / 1,739 |

The crop step enforces **train/serve parity**: training images are cropped with the exact
square-bbox + 25% padding math the browser uses at inference
([crop_dataset.py](crop_dataset.py) `hand_bbox()` mirrors `frontend/js/app.js`).

---

## Experiment A — architecture comparison (uncropped Kaggle, notebook)

Four backbones, transfer-learned on the raw Kaggle images, val split (3,480 eval images).
Source: executed output cells in
[notebooks/ASL_PyTorch_Complete.ipynb](notebooks/ASL_PyTorch_Complete.ipynb).

| Model | Best Val Acc | Params (total) | Trainable | Train time |
|-------|-------------:|---------------:|----------:|-----------:|
| **MobileNetV2** | **99.80%** | 2,261,021 (2.26M) | 2,076,061 | 6.64 min |
| EfficientNet-B0 | 99.77% | 4,044,697 (4.04M) | 3,697,157 | 7.59 min |
| ResNet50 | 99.71% | 23,567,453 (23.57M) | 19,230,237 | 5.36 min |
| CustomCNN (from scratch) | 64.02% | 1,444,541 (1.44M) | 1,444,541 | 11.03 min |

An early external check in the same notebook: the best model scored **99.52%** on the
custom webcam captures vs **99.80%** on Kaggle val — a small, encouraging generalization
gap, though on an easy in-house set (see *Real-world robustness* for the honest caveat).

MobileNetV2 was chosen for deployment: best accuracy at the smallest transfer-learned
footprint and lowest latency.

---

## Experiment B — deployed model (cropped, train/serve parity)

MobileNetV2 retrained on `datasets/combined_cropped` via
[train_and_export.py](train_and_export.py) and exported to ONNX. Source:
[outputs/train.log](outputs/train.log).

| | |
|---|---|
| **Held-out TEST accuracy** | **99.54%** (1,739 images, unseen in train *and* val) |
| Best validation accuracy | 99.77% |
| Split | 80 / 10 / 10 → 13,921 / 1,739 / 1,739, stratified per class, seed 42 |
| Params | 2,261,021 total / 2,245,229 trainable (first 4 feature blocks frozen) |
| Epochs | 15 (AdamW, cosine schedule, label smoothing 0.05) |
| Export | ONNX opset 18, ~9.3 MB weights |

The test set was held out of **both** training and checkpoint selection, so **99.54%** is
an honest generalization estimate — not the val figure the model was tuned against. The
split is verified stratified and leak-free: all 29 classes appear in train/val/test with
zero path overlap. Source: [outputs/train.log](outputs/train.log) +
`outputs/metrics/test_accuracy.txt`.

---

## Deployment & latency

- **Runtime:** ONNX Runtime, CPU (`CPUExecutionProvider`).
- **Model inference latency** (measured by [benchmark.py](benchmark.py), batch 1, warm):
  **~1.1 ms/frame median** (mean 1.2–1.6 ms depending on system load) → **~650–870 FPS** —
  pure `session.run`, this dev machine, multi-threaded ORT.
- **Honest caveat:** that is *model inference only*. End-to-end app latency also includes
  browser capture, client-side MediaPipe hand detection, WebSocket round-trip, and
  server-side preprocessing — all excluded here. The model is not the bottleneck.

Regenerate: `python benchmark.py --model outputs/models/best_mobilenet_v2.onnx --images datasets/combined_cropped`
→ overall accuracy, per-class precision/recall/F1, a confusion-matrix PNG, and latency
(`outputs/metrics/benchmark_*.md` + `.png`).

---

## Real-world robustness

The benchmark can score any labeled folder of **raw** photos, cropping each with the same
MediaPipe pipeline the app uses:

```bash
python benchmark.py --model outputs/models/best_mobilenet_v2.onnx \
    --images datasets/phone_photos --crop
```

**Procedure (to fill this in):** capture ~50 phone photos across varied conditions —
different lighting, cluttered backgrounds, left hand, partial occlusion — into
`datasets/phone_photos/<CLASS>/`, then run the command above. The gap between this number
and the 99.x% val accuracy is the honest measure of the benchmark-vs-reality gap.

> **Status:** harness ready and validated end-to-end; awaiting a captured photo set.
> A benchmark accuracy near 99% on a trivial, class-balanced dataset does **not** imply
> 99% on real webcam input — this section exists to measure that difference, not hide it.

---

## CustomCNN — a deliberate baseline, not a failure

CustomCNN (from scratch) scores **64.02%** vs MobileNetV2's 99.80% on identical data. It
is kept intentionally: the ~35-point gap **quantifies what ImageNet transfer learning
buys** on this task. A from-scratch 1.44M-param net simply does not have the inductive
priors 87K images can instill in 15 epochs. It remains selectable
(`MODEL_TYPE=custom_cnn`) as a reference point, not a recommended model.

---

## Reproduce

```bash
# 1. Crop dataset to hand regions (train/serve parity)
python crop_dataset.py
# 2. Train MobileNetV2 (80/10/10), report held-out test acc, export ONNX
python train_and_export.py
# 3. Benchmark the exported model (accuracy, P/R/F1, confusion matrix, latency)
python benchmark.py --model outputs/models/best_mobilenet_v2.onnx --images datasets/combined_cropped
# 4. Run the test suite (177 checks)
python test_webapp.py && python test_edge_cases.py
```

## Honest limitations

- **Static fingerspelling, not continuous ASL.** The system classifies single hand shapes;
  real ASL is temporal (motion, two hands, facial grammar). A landmark→sequence track is
  scaffolded under `src/scripts/capture_landmark_sequences.py` + `train_sequence.py`.
- **Benchmark dataset is easy** — clean, centered, class-balanced. The 99.54% held-out
  test accuracy is an upper bound on real-world webcam performance, which the
  *Real-world robustness* section exists to measure once phone photos are captured.
