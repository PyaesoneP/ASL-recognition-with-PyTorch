#!/usr/bin/env python3
"""Capture labeled hand-landmark sequences for dynamic-sign recognition (research track).

Desktop webcam tool — run where a camera is available (like ``src/inference``; on WSL,
run it on Windows directly if MediaPipe misbehaves). For each clip it records ``--frames``
frames of 21 (x, y, z) MediaPipe hand landmarks and saves an (T, 63) ``.npy`` under
``datasets/landmark_sequences/<sign>/``, ready for ``train_sequence.py``.

Torch-free (MediaPipe + OpenCV only). Controls: SPACE = record one clip, Q = quit.

Example:
    python src/scripts/capture_landmark_sequences.py --sign hello --clips 30 --frames 45
"""
import argparse
import uuid
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp


def record_clip(cap, hands, n_frames):
    """Record one clip of n_frames landmark vectors. Returns (T, 63) or None if the
    hand was lost for too much of the clip."""
    frames, misses = [], 0
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            frames.append([c for p in lm for c in (p.x, p.y, p.z)])
        else:
            misses += 1
            if frames:
                frames.append(frames[-1])  # hold last pose to keep length stable
        cv2.putText(frame, f"REC {len(frames)}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("capture landmark sequences", frame)
        cv2.waitKey(1)
    if len(frames) < n_frames or misses > n_frames // 3:
        return None
    return np.asarray(frames, dtype=np.float32)  # (T, 63)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", required=True, help="Label for the sign being recorded")
    ap.add_argument("--clips", type=int, default=30, help="How many clips to collect")
    ap.add_argument("--frames", type=int, default=45, help="Frames per clip (~1.5s @ 30fps)")
    ap.add_argument("--out", default="datasets/landmark_sequences", help="Output root")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.sign
    out_dir.mkdir(parents=True, exist_ok=True)

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam (on WSL, run this on Windows directly).")
        return

    saved = 0
    print(f"Recording '{args.sign}': SPACE = record a {args.frames}-frame clip, Q = quit")
    while saved < args.clips:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                                mp.solutions.hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"{args.sign}  saved {saved}/{args.clips}  SPACE=rec  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("capture landmark sequences", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            seq = record_clip(cap, hands, args.frames)
            if seq is not None:
                fp = out_dir / f"{args.sign}_{uuid.uuid4().hex[:8]}.npy"
                np.save(fp, seq)
                saved += 1
                print(f"  saved {fp}  shape={seq.shape}")
            else:
                print("  clip discarded (hand lost for too many frames)")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Done: {saved} clips in {out_dir}")


if __name__ == "__main__":
    main()
