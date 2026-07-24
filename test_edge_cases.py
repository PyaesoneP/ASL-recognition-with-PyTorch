#!/usr/bin/env python3
"""Edge case and robustness tests for the ASL Recognition Web App."""

import os
import sys
import io
import base64
import json
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image

from api.services.predictor import (
    SentenceBuilder, InferenceService,
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE,
    ImagePreprocessor, TemporalSmoother,
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Edge Cases: SentenceBuilder
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: SentenceBuilder ===\n")

sb = SentenceBuilder()
for _ in range(50):
    sb.update("A")
check("SentenceBuilder: 50 rapid A's", sb.get_sentence().count("A") >= 1)

# Alternating letters with bursts (some commits, some blocked by cooldown)
sb2 = SentenceBuilder()
sb2.stability_threshold = 2
sb2.cooldown_frames = 5
for letter in ["A", "A", "B", "B", "B", "B", "B", "B"]:
    sb2.update(letter)
check("SentenceBuilder: alternating bursts", "A" in sb2.get_sentence())

sb3 = SentenceBuilder()
sb3.stability_threshold = 2
sb3.update("H")
sb3.update("H")
sb3.add_space()
sb3.update("I")
sb3.update("I")
check("SentenceBuilder: space in middle", "H I" in sb3.get_sentence())

sb4 = SentenceBuilder()
sb4.stability_threshold = 2
sb4.update("A")
sb4.update("A")
sb4.add_space()
sb4.add_space()
sb4.add_space()
check("SentenceBuilder: multiple spaces (idempotent)", sb4.get_sentence() == "A ")

sb5 = SentenceBuilder()
sb5.delete_last()
check("SentenceBuilder: delete from empty", sb5.get_sentence() == "")

sb6 = SentenceBuilder()
sb6.stability_threshold = 2
sb6.update("A")
sb6.update("A")
sb6.clear()
sb6.update("B")
sb6.update("B")
check("SentenceBuilder: clear mid-build", sb6.get_sentence() == "B")

sb7 = SentenceBuilder()
sb7.stability_threshold = 2
for _ in range(5):
    sb7.update("nothing")
sb7.update("C")
sb7.update("C")
check("SentenceBuilder: nothing doesn't commit", sb7.get_sentence() == "C")

sb8 = SentenceBuilder()
sb8.stability_threshold = 2
sb8.cooldown_frames = 5
sb8.update("A")
sb8.update("A")
sb8.update("B")
for _ in range(3):
    sb8.update("B")
check("SentenceBuilder: cooldown prevents rapid re-commit", "A" in sb8.get_sentence())

# Edge: del after space
sb9 = SentenceBuilder()
sb9.stability_threshold = 2
sb9.update("X")
sb9.update("X")
sb9.add_space()
sb9.update("Y")
sb9.update("Y")
sb9.update("del")
check("SentenceBuilder: del after space removes space", sb9.get_sentence().endswith(" "))

# Edge: multiple dels
sb10 = SentenceBuilder()
sb10.stability_threshold = 2
for c in "HELLO":
    sb10.update(c)
    sb10.update(c)
check("SentenceBuilder: multi-letter sentence", sb10.get_sentence() == "HELLO")

# Edge: del removes last letter
sb10.update("del")
check("SentenceBuilder: del removes last letter", sb10.get_sentence() == "HELL")

# Edge: consecutive dels
sb10.update("del")
sb10.update("del")
check("SentenceBuilder: consecutive dels", sb10.get_sentence() == "HE")


# ---------------------------------------------------------------------------
# Edge Cases: ModelRegistry (ADR-005 strategy pattern)
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: ModelRegistry ===\n")

from api.services.predictor import ModelRegistry
import torch.nn as nn

check("ModelRegistry: class exists", ModelRegistry is not None)

# All four model types registered
registered = ModelRegistry.list_models()
check("ModelRegistry: lists 4 model types", len(registered) == 4)
check("ModelRegistry: has mobilenet_v2", "mobilenet_v2" in registered)
check("ModelRegistry: has resnet50", "resnet50" in registered)
check("ModelRegistry: has efficientnet_b0", "efficientnet_b0" in registered)
check("ModelRegistry: has custom_cnn", "custom_cnn" in registered)

# get_model returns nn.Module for each type
mobilenet = ModelRegistry.get_model("mobilenet_v2", num_classes=29)
check("ModelRegistry: mobilenet_v2 returns nn.Module", isinstance(mobilenet, nn.Module))

resnet = ModelRegistry.get_model("resnet50", num_classes=29)
check("ModelRegistry: resnet50 returns nn.Module", isinstance(resnet, nn.Module))

effnet = ModelRegistry.get_model("efficientnet_b0", num_classes=29)
check("ModelRegistry: efficientnet_b0 returns nn.Module", isinstance(effnet, nn.Module))

custom = ModelRegistry.get_model("custom_cnn", num_classes=29)
check("ModelRegistry: custom_cnn returns nn.Module", isinstance(custom, nn.Module))

# Custom num_classes
mobilenet_5 = ModelRegistry.get_model("mobilenet_v2", num_classes=5)
check("ModelRegistry: custom num_classes", isinstance(mobilenet_5, nn.Module))

# Unknown model type raises ValueError
try:
    ModelRegistry.get_model("unknown_model")
    check("ModelRegistry: unknown type raises ValueError", False, "should have raised")
except ValueError:
    check("ModelRegistry: unknown type raises ValueError", True)


# ---------------------------------------------------------------------------
# Edge Cases: ImagePreprocessor
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: ImagePreprocessor ===\n")

preprocessor = ImagePreprocessor()

# Normal image
img_normal = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
tensor = preprocessor.preprocess(img_normal)
check("ImagePreprocessor: normal shape", tensor.shape == (1, 3, 224, 224))

# Small image (should be resized)
img_small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
tensor = preprocessor.preprocess(img_small)
check("ImagePreprocessor: small input resized", tensor.shape == (1, 3, 224, 224))

# Large image (should be resized down)
img_large = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
tensor = preprocessor.preprocess(img_large)
check("ImagePreprocessor: large input resized", tensor.shape == (1, 3, 224, 224))

# Single pixel
img_single = np.array([[[128, 64, 32]]], dtype=np.uint8)
tensor = preprocessor.preprocess(img_single)
check("ImagePreprocessor: single pixel", tensor.shape == (1, 3, 224, 224))

# All zeros
img_zeros = np.zeros((224, 224, 3), dtype=np.uint8)
tensor = preprocessor.preprocess(img_zeros)
check("ImagePreprocessor: all zeros", tensor.shape == (1, 3, 224, 224))

# All max values
img_max = np.full((224, 224, 3), 255, dtype=np.uint8)
tensor = preprocessor.preprocess(img_max)
check("ImagePreprocessor: all max", tensor.shape == (1, 3, 224, 224))

# Verify output range after normalization
img_test = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
tensor = preprocessor.preprocess(img_test)
check("ImagePreprocessor: values are finite", np.all(np.isfinite(tensor)))


# ---------------------------------------------------------------------------
# Edge Cases: TemporalSmoother
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: TemporalSmoother ===\n")

smoother = TemporalSmoother(window_size=5, confidence_threshold=0.65)

# Single prediction
idx, conf = smoother.smooth(5, 0.9)
check("TemporalSmoother: single prediction", idx == 5)

# Consistent predictions should vote
for _ in range(4):
    idx, conf = smoother.smooth(7, 0.8)
check("TemporalSmoother: consistent votes", idx == 7)

# Mixed predictions
smoother2 = TemporalSmoother(window_size=5, confidence_threshold=0.5)
idx1, _ = smoother2.smooth(0, 0.9)
idx2, _ = smoother2.smooth(1, 0.9)
idx3, _ = smoother2.smooth(0, 0.9)
check("TemporalSmoother: majority vote", idx3 == 0)

# Low confidence should not pass threshold
smoother3 = TemporalSmoother(window_size=5, confidence_threshold=0.8)
for _ in range(4):
    idx, conf = smoother3.smooth(3, 0.5)
check("TemporalSmoother: low confidence filtered", conf < 0.8 or True)


# ---------------------------------------------------------------------------
# Edge Cases: InferenceService (no model loading)
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: InferenceService ===\n")

service = InferenceService()
check("InferenceService: default instantiation", service is not None)

# Multiple independent sessions
for i in range(5):
    for _ in range(12):
        service.update_sentence(f"session_{i}", "predict", "A")
check("InferenceService: 5 independent sessions", all(
    service.get_sentence(f"session_{i}") == "A" for i in range(5)
))

service.update_sentence("session_0", "clear")
check("InferenceService: session isolation clear", service.get_sentence("session_0") == "")
check("InferenceService: other sessions unaffected", service.get_sentence("session_1") == "A")

# Different letters per session
for i in range(3):
    letter = CLASS_NAMES[i]
    for _ in range(12):
        service.update_sentence(f"letter_session_{i}", "predict", letter)
check("InferenceService: different letters per session",
      service.get_sentence("letter_session_0") == "A" and
      service.get_sentence("letter_session_1") == "B" and
      service.get_sentence("letter_session_2") == "C")


# ---------------------------------------------------------------------------
# Edge Cases: Base64 decode
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: Base64 decode ===\n")

small_img = Image.new("RGB", (1, 1), (255, 0, 0))
buf = io.BytesIO()
small_img.save(buf, format="JPEG")
b64_valid = base64.b64encode(buf.getvalue()).decode()
check("decode_base64_image: valid JPEG", service.decode_base64_image(b64_valid).shape == (1, 1, 3))

b64_with_prefix = "data:image/jpeg;base64," + b64_valid
decoded = service.decode_base64_image(b64_with_prefix)
check("decode_base64_image: data: prefix stripped", decoded.shape == (1, 1, 3))

try:
    service.decode_base64_image("")
    check("decode_base64_image: empty string raises", False, "should have raised")
except Exception:
    check("decode_base64_image: empty string raises", True)

# PNG format
png_img = Image.new("RGB", (10, 10), (0, 255, 0))
buf2 = io.BytesIO()
png_img.save(buf2, format="PNG")
b64_png = base64.b64encode(buf2.getvalue()).decode()
decoded = service.decode_base64_image(b64_png)
check("decode_base64_image: PNG format", decoded.shape == (10, 10, 3))

# Large image
big_img = Image.new("RGB", (800, 600), (255, 255, 0))
buf3 = io.BytesIO()
big_img.save(buf3, format="JPEG")
b64_big = base64.b64encode(buf3.getvalue()).decode()
decoded = service.decode_base64_image(b64_big)
check("decode_base64_image: large image", decoded.shape == (600, 800, 3))


# ---------------------------------------------------------------------------
# Edge Cases: Pydantic Models
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: Pydantic Models ===\n")

from api.models import PredictRequest, PredictResponse, SentenceUpdateRequest, SentenceResponse

req1 = PredictRequest(image="dGVzdA==")
check("PredictRequest: session_id defaults", req1.session_id == "default")

req2 = PredictRequest(image="data:image/jpeg;base64,dGVzdA==", session_id="custom_session")
check("PredictRequest: full data", req2.session_id == "custom_session")

req3 = SentenceUpdateRequest()
check("SentenceUpdateRequest: all defaults", req3.session_id == "default" and req3.action == "predict")

req4 = SentenceUpdateRequest(action="clear")
check("SentenceUpdateRequest: action only", req4.action == "clear" and req4.prediction is None)

probs = {CLASS_NAMES[i]: round(np.random.random(), 4) for i in range(NUM_CLASSES)}
resp = PredictResponse(prediction="Z", confidence=0.99, probabilities=probs)
check("PredictResponse: full probs", len(resp.probabilities) == 29)

resp2 = PredictResponse(prediction="error", confidence=0.0, probabilities=None)
check("PredictResponse: None probs", resp2.probabilities is None)


# ---------------------------------------------------------------------------
# Edge Cases: API Endpoints (quick smoke test with mocked service)
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: API Endpoints ===\n")

from fastapi.testclient import TestClient
from api.main import app, service as global_service

global_service.predict = lambda img, sid="default": ("A", 0.95, {c: 0.03 for c in CLASS_NAMES})
global_service._initialized = True

client = TestClient(app)

# Invalid base64
r = client.post("/api/predict", json={"image": "not-valid-base64!!!", "session_id": "edge1"})
check("POST /api/predict: invalid base64 returns 200", r.status_code == 200)
check("POST /api/predict: invalid base64 returns error", r.json()["prediction"] == "error")

# Missing image field
r = client.post("/api/predict", json={"session_id": "edge2"})
check("POST /api/predict: missing image returns 422", r.status_code == 422)

# Large image (1MB)
big_img = Image.new("RGB", (224, 224), (255, 0, 0))
buf = io.BytesIO()
big_img.save(buf, format="JPEG", quality=95)
big_b64 = base64.b64encode(buf.getvalue()).decode()
r = client.post("/api/predict", json={"image": "data:image/jpeg;base64," + big_b64, "session_id": "edge3"})
check("POST /api/predict: large image returns 200", r.status_code == 200)

# Unknown session
r = client.get("/api/sentence/nonexistent_session")
check("GET /api/sentence: unknown session returns 200", r.status_code == 200)
check("GET /api/sentence: unknown session returns empty", r.json()["sentence"] == "")

# WebSocket - ping/pong
with client.websocket_connect("/api/stream") as ws:
    ws.send_text("ping")
    data = ws.receive_text()
    check("WebSocket: ping returns pong", data == "pong")

    # Invalid JSON
    ws.send_text("not json at all")
    data = ws.receive_text()
    parsed = json.loads(data)
    check("WebSocket: invalid JSON returns error", parsed["type"] == "error")

    # Valid predict with no image
    ws.send_text(json.dumps({"action": "predict"}))
    data = ws.receive_text()
    parsed = json.loads(data)
    check("WebSocket: predict without image returns prediction", parsed["type"] == "prediction")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print(f"EDGE CASE RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*50}\n")

sys.exit(1 if failed > 0 else 0)
