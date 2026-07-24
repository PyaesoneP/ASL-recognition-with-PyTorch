#!/usr/bin/env python3
"""Train MobileNetV2 on ASL dataset and export to ONNX for deployment."""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────
TRAIN_DIR = "datasets/combined_cropped"   # MediaPipe hand crops — matches the live app
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05    # light — keeps calibration without capping confidence too hard
DROPOUT = 0.3
FREEZE_BLOCKS = 4          # freeze the first N of MobileNetV2's 19 feature blocks
VAL_FRACTION = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CLASS_NAMES = [c for c in sorted(os.listdir(TRAIN_DIR)) if os.path.isdir(os.path.join(TRAIN_DIR, c))]
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# ── Data transforms ─────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ASLDataset(torch.utils.data.Dataset):
    """Serves a fixed list of (path, label) samples with a given transform."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def gather_samples(root_dir):
    """Collect (path, label) pairs grouped per class for a stratified split."""
    root = Path(root_dir)
    per_class = {i: [] for i in range(NUM_CLASSES)}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_path = root / cls_name
        if not cls_path.exists():
            continue
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                per_class[cls_idx].append((str(cls_path / fname), cls_idx))
    return per_class


# ── Stratified train/val split ──────────────────────────────────────────
# Bug fix: the previous code took the first 85%% / last 15%% of a list that was
# ordered class-by-class, so entire classes landed in val and were never
# trained. We now split WITHIN each class so both sets cover all 29 classes.
per_class = gather_samples(TRAIN_DIR)
train_samples, val_samples = [], []
rng = random.Random(SEED)
for cls_idx, items in per_class.items():
    rng.shuffle(items)
    # Guarantee at least one val sample for any class with >=2 images, and
    # never take them all, so every class is represented in both splits.
    n_val = max(1, int(len(items) * VAL_FRACTION)) if len(items) > 1 else 0
    val_samples.extend(items[:n_val])
    train_samples.extend(items[n_val:])
rng.shuffle(train_samples)

train_ds = ASLDataset(train_samples, transform=train_transform)
val_ds = ASLDataset(val_samples, transform=val_transform)   # Bug fix: no augmentation on val
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

pin = DEVICE.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=pin, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=pin, persistent_workers=True)

# ── Model ───────────────────────────────────────────────────────────────
print("Loading MobileNetV2 (ImageNet pretrained)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.last_channel, NUM_CLASSES),
)
model = model.to(DEVICE)

# Bug fix: freeze the first N feature blocks by INDEX. The old `"0" in name`
# substring check matched blocks 10-18 too and froze almost the whole network.
for idx, block in enumerate(model.features):
    if idx < FREEZE_BLOCKS:
        for param in block.parameters():
            param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,} / {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

os.makedirs("outputs/models", exist_ok=True)
best_pth = "outputs/models/best_mobilenet_v2.pth"

# ── Training loop ───────────────────────────────────────────────────────
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = correct = seen = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += images.size(0)

    train_acc = correct / seen
    train_loss = running_loss / seen

    # Validation
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_pth)
        print(f"  -> New best model saved (val acc: {best_val_acc:.4f})")

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# ── Export to ONNX ──────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_pth, weights_only=True))
model.to("cpu").eval()   # Bug fix: model + dummy_input must be on the same device

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
output_path = "outputs/models/best_mobilenet_v2.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"ONNX model saved to {output_path} ({size_mb:.1f} MB)")
print("Done!")
