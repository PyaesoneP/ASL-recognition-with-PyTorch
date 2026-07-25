#!/usr/bin/env python3
"""Benchmark an ONNX ASL model over a labeled image folder.

Reports overall accuracy, per-class precision / recall / F1, a confusion-matrix
PNG, and inference latency (ms/frame + FPS). One command, full report — the
number that turns a demo into an evaluated system.

The script is intentionally **torch-free** (ONNX Runtime only). That keeps the
optional ``--crop`` path — which loads MediaPipe to hand-crop raw photos — in the
same process without the torch<->mediapipe native-library segfault seen on some
Linux/WSL setups. Constants come from the single source of truth
(``src/config/settings.py``) and the preprocessing mirrors the live API's
``ImagePreprocessor`` exactly, so the measured accuracy reflects what the
deployed model actually sees.

Usage
-----
  # In-distribution test set (already hand-cropped, e.g. a held-out split):
  python benchmark.py --model outputs/models/best_mobilenet_v2.onnx \
      --images datasets/combined_cropped

  # Raw phone photos (full frames) — crop with MediaPipe first, like the app:
  python benchmark.py --model outputs/models/best_mobilenet_v2.onnx \
      --images datasets/phone_photos --crop

``--images`` must contain one subdirectory per class (named A, B, ..., del,
nothing, space); files in unrecognised subdirectories are ignored.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Single source of truth for the label set (torch-free import).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.config.settings import CLASS_NAMES, NUM_CLASSES  # noqa: E402

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
_IMG_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Preprocessing — mirrors api/services/predictor.py ImagePreprocessor exactly
# (BGR->RGB, PIL resize, /255, ImageNet normalise, NCHW) so eval == serving.
# ---------------------------------------------------------------------------
def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = image_bgr[:, :, ::-1].copy()
    image_resized = Image.fromarray(image_rgb).resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image_resized).astype(np.float32).transpose(2, 0, 1)
    arr /= 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0)


def gather_samples(images_dir: Path, limit_per_class: int | None):
    """Collect (path, label_idx) for every image under a recognised class dir."""
    samples = []
    skipped_dirs = []
    for sub in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        if sub.name not in _CLASS_TO_IDX:
            skipped_dirs.append(sub.name)
            continue
        label = _CLASS_TO_IDX[sub.name]
        files = sorted(f for f in os.listdir(sub) if f.lower().endswith(_IMG_EXTS))
        if limit_per_class:
            files = files[:limit_per_class]
        samples.extend((sub / f, label) for f in files)
    return samples, skipped_dirs


def crop_hand(bgr: np.ndarray, landmarker, hand_bbox) -> np.ndarray:
    """Hand-crop a full frame the same way crop_dataset.py / the frontend do.
    Falls back to the whole frame when no hand is detected (e.g. 'nothing')."""
    import mediapipe as mp
    vh, vw = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if res.hand_landmarks:
        x, y, w, h = hand_bbox(res.hand_landmarks[0], vw, vh)
        if w > 0 and h > 0:
            return bgr[y:y + h, x:x + w]
    return bgr


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (confusion_matrix, per_class dict, overall_acc, macro dict).
    Pure numpy — no scikit-learn dependency."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        support = cm[i, :].sum()
        pred_pos = cm[:, i].sum()
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = dict(precision=precision, recall=recall, f1=f1, support=int(support))
        if support:  # only average over classes present in the eval set
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    overall_acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
    macro = dict(
        precision=float(np.mean(precisions)) if precisions else 0.0,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        f1=float(np.mean(f1s)) if f1s else 0.0,
    )
    return cm, per_class, float(overall_acc), macro


def save_confusion_png(cm: np.ndarray, out_path: Path, title: str):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised frequency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="outputs/models/best_mobilenet_v2.onnx",
                    help="Path to the .onnx model (default: %(default)s)")
    ap.add_argument("--images", required=True,
                    help="Folder with one subdirectory per class")
    ap.add_argument("--crop", action="store_true",
                    help="MediaPipe hand-crop each image first (for raw/full-frame photos)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap images per class (for a quick smoke run)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Warm-up inferences before timing (default: %(default)s)")
    ap.add_argument("--output-dir", default="outputs/metrics",
                    help="Where to write the confusion matrix + report (default: %(default)s)")
    args = ap.parse_args()

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        ap.error(f"--images '{images_dir}' is not a directory")
    if not Path(args.model).exists():
        ap.error(f"--model '{args.model}' not found")

    samples, skipped = gather_samples(images_dir, args.limit)
    if not samples:
        ap.error(f"No class subdirectories with images found under '{images_dir}'. "
                 f"Expected dirs named like {CLASS_NAMES[:3]}...")
    if skipped:
        print(f"[!] Ignored {len(skipped)} unrecognised subdir(s): {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))

    print(f"Model:  {args.model}")
    print(f"Images: {images_dir}  ({len(samples)} images, crop={args.crop})")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    landmarker = hand_bbox = None
    if args.crop:
        from crop_dataset import build_landmarker, hand_bbox as _hand_bbox  # lazy: loads mediapipe
        landmarker, hand_bbox = build_landmarker(), _hand_bbox

    # Warm up (first ORT run pays graph-init / allocation costs).
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for _ in range(max(0, args.warmup)):
        sess.run(None, {input_name: dummy})

    y_true, y_pred, latencies_ms = [], [], []
    unreadable = 0
    for path, label in samples:
        bgr = cv2.imread(str(path))
        if bgr is None:
            unreadable += 1
            continue
        if args.crop:
            bgr = crop_hand(bgr, landmarker, hand_bbox)
        batch = preprocess(bgr)
        t0 = time.perf_counter()
        logits = sess.run(None, {input_name: batch})[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        y_true.append(label)
        y_pred.append(int(np.argmax(logits[0])))

    if not y_pred:
        ap.error("No images could be read.")
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm, per_class, overall_acc, macro = confusion_and_metrics(y_true, y_pred)
    lat = np.array(latencies_ms)

    # ---- Report ----------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"Evaluated {len(y_pred)} images"
          + (f"  ({unreadable} unreadable, skipped)" if unreadable else ""))
    print(f"Overall accuracy : {overall_acc * 100:.2f}%")
    print(f"Macro precision  : {macro['precision'] * 100:.2f}%")
    print(f"Macro recall     : {macro['recall'] * 100:.2f}%")
    print(f"Macro F1         : {macro['f1'] * 100:.2f}%")
    print(f"Latency / frame  : mean {lat.mean():.2f} ms | p50 {np.percentile(lat,50):.2f} | "
          f"p95 {np.percentile(lat,95):.2f}  ->  {1000.0/lat.mean():.1f} FPS")
    print("=" * 64)

    worst = sorted((m["f1"], n, m) for n, m in per_class.items() if m["support"])[:8]
    print("\nWeakest classes by F1:")
    print(f"  {'class':<8} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}")
    for _, name, m in worst:
        print(f"  {name:<8} {m['precision']*100:6.1f} {m['recall']*100:7.1f} "
              f"{m['f1']*100:6.1f} {m['support']:8d}")

    out_dir = Path(args.output_dir)
    stem = Path(args.model).stem + ("_cropped" if args.crop else "")
    png_path = out_dir / f"benchmark_confusion_{stem}.png"
    save_confusion_png(cm, png_path, title=f"{Path(args.model).name} — {images_dir.name}")
    print(f"\nConfusion matrix saved: {png_path}")

    # Markdown summary for RESULTS.md to cite.
    md_path = out_dir / f"benchmark_{stem}.md"
    with open(md_path, "w") as f:
        f.write(f"# Benchmark: {Path(args.model).name} on `{images_dir}`\n\n")
        f.write(f"- Images: **{len(y_pred)}** (crop={args.crop})\n")
        f.write(f"- Overall accuracy: **{overall_acc*100:.2f}%**\n")
        f.write(f"- Macro precision / recall / F1: "
                f"{macro['precision']*100:.2f}% / {macro['recall']*100:.2f}% / {macro['f1']*100:.2f}%\n")
        f.write(f"- Latency: mean **{lat.mean():.2f} ms** "
                f"(p50 {np.percentile(lat,50):.2f}, p95 {np.percentile(lat,95):.2f}), "
                f"**{1000.0/lat.mean():.1f} FPS** (CPU, ONNX Runtime)\n\n")
        f.write("| class | precision | recall | f1 | support |\n|---|---|---|---|---|\n")
        for name in CLASS_NAMES:
            m = per_class[name]
            f.write(f"| {name} | {m['precision']*100:.1f} | {m['recall']*100:.1f} "
                    f"| {m['f1']*100:.1f} | {m['support']} |\n")
    print(f"Markdown report saved: {md_path}")


if __name__ == "__main__":
    main()
