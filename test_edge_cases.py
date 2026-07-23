#!/usr/bin/env python3
"""Edge case and robustness tests for the ASL Recognition Web App."""

import os
import sys
import io
import base64
import json
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from api.services.predictor import (
    ASLPredictor, SentenceBuilder, InferenceService,
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE,
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


def make_dummy_model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=None, progress=False)
    model.classifier[1] = nn.Linear(model.last_channel, 29)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
    return model


# ---------------------------------------------------------------------------
# Edge Cases: Predictor
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: Predictor ===\n")

dummy = make_dummy_model()
predictor = ASLPredictor(dummy, "mobilenet_v2")

check("Predict with None image", predictor.predict(None) == (None, None, None))
check("Predict with zero array", predictor.predict(np.zeros((224, 224, 3), dtype=np.uint8))[0] in CLASS_NAMES)
check("Predict with random array", predictor.predict(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))[0] in CLASS_NAMES)
check("Predict with 32x32 image", predictor.predict(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))[0] in CLASS_NAMES)
check("Predict with 1x1 image", predictor.predict(np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8))[0] in CLASS_NAMES)
check("Predict with grayscale image", predictor.predict(np.random.randint(0, 255, (224, 224), dtype=np.uint8)) == (None, None, None))
check("Predict with 4K image", predictor.predict(np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8))[0] in CLASS_NAMES)
single_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
check("Predict with single pixel", predictor.predict(single_pixel)[0] in CLASS_NAMES)

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

# ---------------------------------------------------------------------------
# Edge Cases: InferenceService
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: InferenceService ===\n")

service = InferenceService()
service.model = dummy
service.predictor = predictor
service._initialized = True

for i in range(5):
    for _ in range(12):
        service.update_sentence(f"session_{i}", "predict", "A")
check("InferenceService: 5 independent sessions", all(
    service.get_sentence(f"session_{i}") == "A" for i in range(5)
))

service.update_sentence("session_0", "clear")
check("InferenceService: session isolation", service.get_sentence("session_0") == "")
check("InferenceService: other sessions unaffected", service.get_sentence("session_1") == "A")

# Base64 edge cases
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
# Edge Cases: API Endpoints (quick smoke test)
# ---------------------------------------------------------------------------

print("\n=== EDGE CASES: API Endpoints ===\n")

from fastapi.testclient import TestClient
from api.main import app, service as global_service

global_service.model = dummy
global_service.predictor = ASLPredictor(dummy, "mobilenet_v2")
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
