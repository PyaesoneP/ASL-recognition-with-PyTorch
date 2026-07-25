#!/usr/bin/env python3
"""Train a dynamic-sign classifier on captured landmark sequences (research track).

Loads (T, 63) ``.npy`` clips from ``datasets/landmark_sequences/<sign>/``, normalizes
them, splits **80/10/10 stratified per class**, trains the GRU in
[src/models_seq.py](src/models_seq.py), and reports a **held-out TEST accuracy** — the
same rigor as [train_and_export.py](train_and_export.py), carried into the temporal track
from day one.

Collect data first with ``src/scripts/capture_landmark_sequences.py``.

Example:
    python train_sequence.py --data datasets/landmark_sequences --epochs 40
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src.models_seq import SignSequenceModel, normalize_landmarks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def pad_or_truncate(seq: np.ndarray, T: int) -> np.ndarray:
    if len(seq) >= T:
        return seq[:T]
    pad = np.repeat(seq[-1:], T - len(seq), axis=0)
    return np.concatenate([seq, pad], axis=0)


class ClipDataset(Dataset):
    def __init__(self, samples, T):
        self.samples = samples
        self.T = T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        seq = pad_or_truncate(normalize_landmarks(np.load(path)), self.T)
        return torch.from_numpy(seq).float(), label


def stratified_split(per_class, seed=SEED):
    """80/10/10 within each class so every sign appears in train/val/test, no overlap."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in per_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * 0.10)) if n > 2 else 0
        n_val = max(1, int(n * 0.10)) if n > 2 else 0
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/landmark_sequences")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    root = Path(args.data)
    signs = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if not signs:
        print(f"No sign folders under {root}. Capture clips first with "
              f"src/scripts/capture_landmark_sequences.py")
        return
    label_map = {s: i for i, s in enumerate(signs)}

    per_class = {i: [] for i in range(len(signs))}
    for s in signs:
        for f in sorted(os.listdir(root / s)):
            if f.endswith(".npy"):
                per_class[label_map[s]].append((str(root / s / f), label_map[s]))

    train, val, test = stratified_split(per_class)
    print(f"signs = {signs}")
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")
    if not train:
        print("Not enough clips to train (need >2 per sign).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader(ds, shuffle):
        return DataLoader(ClipDataset(ds, args.frames), batch_size=args.batch, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader(train, True), loader(val, False), loader(test, False)

    model = SignSequenceModel(num_classes=len(signs)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(dl):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        return correct / total if total else 0.0

    os.makedirs("outputs/models", exist_ok=True)
    best_path = "outputs/models/best_sign_gru.pt"
    best_val = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        va = evaluate(val_loader)
        if va >= best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(), "signs": signs}, best_path)
        print(f"epoch {ep:2d}/{args.epochs}  val_acc {va:.3f}")

    if test:
        model.load_state_dict(torch.load(best_path)["state_dict"])
        print(f"\nBest val {best_val:.3f} | Held-out TEST {evaluate(test_loader):.3f} "
              f"| model -> {best_path}")


if __name__ == "__main__":
    main()
