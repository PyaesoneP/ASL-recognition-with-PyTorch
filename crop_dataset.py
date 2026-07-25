#!/usr/bin/env python3
"""Pre-crop the ASL dataset to MediaPipe hand regions so training matches the
live app. The bbox math mirrors frontend/js/app.js getBounds()/cropAndPredict()
exactly: square box around the 21 hand landmarks, 25%% padding, clamped to the
frame, then resized to 224x224. Images with no detected hand fall back to a
full-frame resize (e.g. the 'nothing' class) so no class loses samples."""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

SRC = Path("datasets/combined_training")
DST = Path("datasets/combined_cropped")
IMG_SIZE = 224
PADDING = 0.25          # must match frontend getBounds()
MODEL = "outputs/mp/hand_landmarker.task"


def build_landmarker(model_path=MODEL):
    """Create a single-hand MediaPipe HandLandmarker (IMAGE mode).

    A factory rather than module-level state, so this module can be imported for
    its pure helpers (hand_bbox) without needing the .task file present.
    benchmark.py reuses build_landmarker()/hand_bbox() for its --crop path.
    """
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def hand_bbox(landmarks, vw, vh):
    """Replicates getBounds() + cropAndPredict() clamping. Returns pixel x,y,w,h."""
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    size = max(max_x - min_x, max_y - min_y) * (1 + PADDING)
    bx, by = cx - size / 2, cy - size / 2

    x = max(0, int(bx * vw))
    y = max(0, int(by * vh))
    w = min(vw - x, int(size * vw))
    h = min(vh - y, int(size * vh))
    return x, y, w, h


def main():
    landmarker = build_landmarker()
    classes = sorted(c.name for c in SRC.iterdir() if c.is_dir())
    total = detected = fallback = 0
    per_class = {}

    for cls in classes:
        (DST / cls).mkdir(parents=True, exist_ok=True)
        d_hit = d_miss = 0
        for fname in sorted(os.listdir(SRC / cls)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total += 1
            bgr = cv2.imread(str(SRC / cls / fname))
            if bgr is None:
                continue
            vh, vw = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            if res.hand_landmarks:
                x, y, w, h = hand_bbox(res.hand_landmarks[0], vw, vh)
                if w > 0 and h > 0:
                    crop = bgr[y:y + h, x:x + w]
                    d_hit += 1
                    detected += 1
                else:
                    crop = bgr
                    d_miss += 1
                    fallback += 1
            else:
                crop = bgr            # fallback: whole frame (e.g. 'nothing')
                d_miss += 1
                fallback += 1

            out = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(DST / cls / fname), out)

        per_class[cls] = (d_hit, d_hit + d_miss)
        print(f"{cls:8s}: {d_hit}/{d_hit + d_miss} hand-cropped", flush=True)

    print(f"\nTotal {total} | hand-cropped {detected} "
          f"({100*detected/max(total,1):.1f}%) | full-frame fallback {fallback}")
    print("Low-detection classes (<80%):",
          [c for c, (h, n) in per_class.items() if n and h / n < 0.8])


if __name__ == "__main__":
    main()
