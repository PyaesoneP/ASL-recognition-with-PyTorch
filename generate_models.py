#!/usr/bin/env python3
"""Generate minimal valid model files for docker-compose testing."""
import os
import torch
import torch.nn as nn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODEL_DIR = "outputs/models"
IMG_SIZE = 224
NUM_CLASSES = 29

def save_mobilenet_v2():
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_mobilenet_v2.pth"))
    print(f"Saved {MODEL_DIR}/best_mobilenet_v2.pth ({os.path.getsize(os.path.join(MODEL_DIR, 'best_mobilenet_v2.pth'))} bytes)")

def save_resnet50():
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_resnet50.pth"))
    print(f"Saved {MODEL_DIR}/best_resnet50.pth ({os.path.getsize(os.path.join(MODEL_DIR, 'best_resnet50.pth'))} bytes)")

def save_efficientnet_b0():
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_efficientnet_b0.pth"))
    print(f"Saved {MODEL_DIR}/best_efficientnet_b0.pth ({os.path.getsize(os.path.join(MODEL_DIR, 'best_efficientnet_b0.pth'))} bytes)")

def save_custom_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, NUM_CLASSES),
    )
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_custom_cnn.pth"))
    print(f"Saved {MODEL_DIR}/best_custom_cnn.pth ({os.path.getsize(os.path.join(MODEL_DIR, 'best_custom_cnn.pth'))} bytes)")

if __name__ == "__main__":
    print("Generating model files...")
    save_mobilenet_v2()
    save_resnet50()
    save_efficientnet_b0()
    save_custom_cnn()
    print("Done.")
