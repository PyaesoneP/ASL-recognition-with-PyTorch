#!/usr/bin/env python3
"""Comprehensive tests for the ASL Recognition Web App."""

import os
import sys
import json
import base64
import io
import tempfile

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers — create a minimal model so we can test inference without LFS files
# ---------------------------------------------------------------------------

def create_dummy_mobilenet():
    """Create a tiny MobileNetV2 with 29 classes for testing."""
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", weights=None, progress=False)
    model.classifier[1] = nn.Linear(model.last_channel, 29)
    # Initialize with small random weights
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
    return model


def create_dummy_cnn():
    """Create a tiny CustomCNN for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 29),
    )
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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


# ---- Phase 1: Backend ----

print("\n=== PHASE 1: Backend API ===\n")

# 1a. Models
print("[1a] Pydantic models")
try:
    from api.models import (
        PredictRequest, PredictResponse,
        SentenceUpdateRequest, SentenceResponse,
        HealthResponse,
    )
    check("Import all models", True)

    req = PredictRequest(image="data:image/jpeg;base64,abc123", session_id="test1")
    check("PredictRequest validation", req.image == "data:image/jpeg;base64,abc123" and req.session_id == "test1")

    resp = PredictResponse(prediction="A", confidence=0.95, probabilities={"A": 0.95})
    check("PredictResponse validation", resp.prediction == "A" and resp.confidence == 0.95)

    sreq = SentenceUpdateRequest(session_id="s1", action="space")
    check("SentenceUpdateRequest validation", sreq.action == "space")

    sresp = SentenceResponse(sentence="hello world", session_id="s1")
    check("SentenceResponse validation", sresp.sentence == "hello world")

    hresp = HealthResponse(status="ok", model_type="mobilenet_v2", model_loaded=True)
    check("HealthResponse validation", hresp.status == "ok")
except Exception as e:
    check("Import all models", False, str(e))

# 1b. Predictor service
print("\n[1b] Predictor service (dummy model)")
try:
    from api.services.predictor import (
        CLASS_NAMES, NUM_CLASSES, IMG_SIZE,
        ASLPredictor, SentenceBuilder, InferenceService, load_model,
        CustomCNN, SMOOTHING_WINDOW, CONFIDENCE_THRESHOLD,
    )
    check("Import predictor module", True)
    check("CLASS_NAMES has 29 classes", len(CLASS_NAMES) == 29)
    check("NUM_CLASSES == 29", NUM_CLASSES == 29)
    check("IMG_SIZE == 224", IMG_SIZE == 224)
    check("CLASS_NAMES includes 'del','nothing','space'",
          "del" in CLASS_NAMES and "nothing" in CLASS_NAMES and "space" in CLASS_NAMES)
    check("CustomCNN class exists", CustomCNN is not None)

    # Load dummy model
    dummy = create_dummy_mobilenet()
    predictor = ASLPredictor(dummy, "mobilenet_v2")
    check("ASLPredictor instantiation", predictor is not None)

    # Predict on random image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    label, conf, probs = predictor.predict(img)
    check("Predict returns label", label is not None and label in CLASS_NAMES)
    check("Predict returns confidence 0-1", 0 <= conf <= 1)
    check("Predict returns prob dict", isinstance(probs, dict) and len(probs) == 29)

    # Predict on error input
    label2, conf2, probs2 = predictor.predict(None)
    check("Predict handles None gracefully", label2 is None and conf2 is None)

    # SentenceBuilder
    sb = SentenceBuilder()
    check("SentenceBuilder init", sb.get_sentence() == "")

    # Add letters with stability
    for _ in range(12):
        sb.update("A")
    check("SentenceBuilder adds letter after stability", "A" in sb.get_sentence())

    # Space
    sb.add_space()
    check("SentenceBuilder adds space", sb.get_sentence().endswith(" "))

    # Delete
    sb.delete_last()
    check("SentenceBuilder deletes last", not sb.get_sentence().endswith(" "))

    # Clear
    sb.clear()
    check("SentenceBuilder clears", sb.get_sentence() == "")

    # Nothing filter
    sb2 = SentenceBuilder()
    for _ in range(12):
        sb2.update("nothing")
    check("SentenceBuilder filters 'nothing'", sb2.get_sentence() == "")

    # InferenceService
    service = InferenceService(model_path="outputs/models/best_mobilenet_v2.pth", model_type="mobilenet_v2")
    service.model = dummy
    service.predictor = predictor
    service._initialized = True

    label3, conf3, _ = service.predict(img, "session1")
    check("InferenceService.predict", label3 is not None)

    # Simulate stability threshold: need 12 consecutive same predictions to commit
    for _ in range(12):
        service.update_sentence("session1", "predict", "A")
    sent = service.get_sentence("session1")
    check("InferenceService.update_sentence (after stability)", sent == "A")

    sent = service.get_sentence("session1")
    check("InferenceService.get_sentence", sent == "A")

    sent = service.update_sentence("session1", "space")
    check("InferenceService space action", sent == "A ")

    sent = service.update_sentence("session1", "del")
    check("InferenceService del action", sent == "A")

    sent = service.update_sentence("session1", "clear")
    check("InferenceService clear action", sent == "")

    # Base64 decode
    pil_img = Image.new("RGB", (224, 224), (128, 64, 32))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    decoded = service.decode_base64_image("data:image/jpeg;base64," + b64)
    check("decode_base64_image", decoded.shape == (224, 224, 3))

    # New architectural components
    from api.services.predictor import (
        ImagePreprocessor, TemporalSmoother, ModelRegistry,
    )
    check("ImagePreprocessor class exists", ImagePreprocessor is not None)
    check("TemporalSmoother class exists", TemporalSmoother is not None)
    check("ModelRegistry class exists", ModelRegistry is not None)

    # ImagePreprocessor
    preprocessor = ImagePreprocessor()
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess(test_img)
    check("ImagePreprocessor outputs (1,3,224,224) tensor",
          tensor.shape == (1, 3, 224, 224))

    # TemporalSmoother
    smoother = TemporalSmoother(window_size=5, confidence_threshold=0.65)
    idx, conf = smoother.smooth(0, 0.9)
    check("TemporalSmoother smooth returns idx and conf", idx is not None and conf is not None)

    # ModelRegistry strategy pattern
    reg_model = ModelRegistry.get_model("mobilenet_v2", num_classes=29)
    check("ModelRegistry.get_model returns nn.Module", isinstance(reg_model, nn.Module))
    reg_model = ModelRegistry.get_model("resnet50", num_classes=29)
    check("ModelRegistry supports resnet50", isinstance(reg_model, nn.Module))
    reg_model = ModelRegistry.get_model("efficientnet_b0", num_classes=29)
    check("ModelRegistry supports efficientnet_b0", isinstance(reg_model, nn.Module))
    reg_model = ModelRegistry.get_model("custom_cnn", num_classes=29)
    check("ModelRegistry supports custom_cnn", isinstance(reg_model, nn.Module))

    # load_model backward compat wrapper
    compat_model = load_model("outputs/models/best_mobilenet_v2.pth")
    check("load_model() backward compat wrapper works", compat_model is not None)

except Exception as e:
    import traceback
    check("Predictor service tests", False, traceback.format_exc())

# 1c. FastAPI app
print("\n[1c] FastAPI app")
try:
    from api.main import app, service as global_service
    check("FastAPI app import", True)
    check("App title", app.title == "ASL Recognition API")

    # Initialize global service with dummy model BEFORE creating TestClient
    # (TestClient runs lifespan which tries to load the real model)
    dummy_model = create_dummy_mobilenet()
    test_predictor = ASLPredictor(dummy_model, "mobilenet_v2")
    global_service.model = dummy_model
    global_service.predictor = test_predictor
    global_service._initialized = True

    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Health
    r = client.get("/api/health")
    check("GET /api/health returns 200", r.status_code == 200)
    health = r.json()
    check("Health response structure", "status" in health and "model_type" in health)

    # Predict (with dummy image)
    pil_img = Image.new("RGB", (224, 224), (100, 150, 200))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    r = client.post("/api/predict", json={"image": "data:image/jpeg;base64," + b64, "session_id": "t1"})
    check("POST /api/predict returns 200", r.status_code == 200)
    pred = r.json()
    check("Predict response has prediction", "prediction" in pred)
    check("Predict response has confidence", "confidence" in pred)
    check("Predict returns valid class", pred["prediction"] in CLASS_NAMES)

    # Sentence update
    for _ in range(12):
        r = client.post("/api/sentence/update", json={"session_id": "t1", "action": "predict", "prediction": "A"})
    r = client.get("/api/sentence/t1")
    check("POST /api/sentence/update works", r.status_code == 200)
    check("Sentence accumulation via API", r.json()["sentence"] == "A")

    # Space
    r = client.post("/api/sentence/update", json={"session_id": "t1", "action": "space"})
    check("Space action via API", r.json()["sentence"] == "A ")

    # Del
    r = client.post("/api/sentence/update", json={"session_id": "t1", "action": "del"})
    check("Del action via API", r.json()["sentence"] == "A")

    # Clear
    r = client.post("/api/sentence/update", json={"session_id": "t1", "action": "clear"})
    check("Clear action via API", r.json()["sentence"] == "")

    # WebSocket
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text("ping")
        # TestClient may return empty control frame first; retry once
        data_raw = ws.receive_text()
        if not data_raw:
            data_raw = ws.receive_text()
        # "pong" is sent as plain string, not JSON
        check("WebSocket ping/pong", data_raw == "pong", f"received: {repr(data_raw)[:50]}")

        ws.send_text(json.dumps({"action": "predict", "image": "data:image/jpeg;base64," + b64}))
        try:
            data = json.loads(ws.receive_text())
            check("WebSocket prediction message", data.get("type") == "prediction")
        except Exception as e:
            check("WebSocket prediction message", False, str(e)[:100])

    # /api/update endpoint (documented contract)
    r = client.post("/api/update", json={
        "class_label": "A",
        "confidence": 0.9,
        "session_id": "update_test",
    })
    check("POST /api/update returns 200", r.status_code == 200)
    update_resp = r.json()
    check("/api/update has sentence field", "sentence" in update_resp)
    check("/api/update has added_letter field", "added_letter" in update_resp)
    check("/api/update has session_id field", "session_id" in update_resp)
    check("/api/update returns added_letter='A'", update_resp.get("added_letter") == "A")

    # /api/update with special actions
    r = client.post("/api/update", json={
        "class_label": "space",
        "confidence": 0.9,
        "session_id": "update_test",
    })
    check("/api/update with 'space' action", r.status_code == 200)

    check("All API endpoint tests", True)

except ImportError:
    check("FastAPI test client", False, "fastapi.testclient not available — install fastapi[test]")
except Exception as e:
    import traceback
    check("FastAPI app tests", False, traceback.format_exc()[:300])

# ---- Phase 2: Frontend ----

print("\n=== PHASE 2: Frontend ===\n")

# 2a. HTML structure
print("[2a] HTML structure")
try:
    with open("frontend/index.html") as f:
        html = f.read()
    check("HTML has video element", "<video" in html and 'id="video"' in html)
    check("HTML has overlay canvas", "<canvas" in html and 'id="overlay"' in html)
    check("HTML has sentence box", 'id="sentenceBox"' in html)
    check("HTML has prediction label", 'id="predictionLabel"' in html)
    check("HTML has control buttons", 'id="btnClear"' in html and 'id="btnSpace"' in html and 'id="btnDel"' in html)
    check("HTML loads CSS", 'href="css/style.css"' in html)
    check("HTML loads sentence.js", 'src="js/sentence.js"' in html)
    check("HTML loads app.js", 'src="js/app.js"' in html)
    check("HTML loads MediaPipe Hands", "@mediapipe/hands/hands.js" in html)
    check("HTML loads MediaPipe Camera", "@mediapipe/camera_utils/camera_utils.js" in html)
    check("HTML has viewport meta", 'name="viewport"' in html)
except Exception as e:
    check("HTML structure", False, str(e))

# 2b. CSS
print("\n[2b] CSS")
try:
    with open("frontend/css/style.css") as f:
        css = f.read()
    check("CSS has dark background", "#0f0f1a" in css)
    check("CSS has video-container", ".video-container" in css)
    check("CSS has sentence-box", ".sentence-box" in css)
    check("CSS has prediction-label", ".prediction-label" in css)
    check("CSS has confidence-bar", ".confidence-bar" in css)
    check("CSS has responsive breakpoint", "@media" in css and "768px" in css)
    check("CSS has ROI box", ".roi-box" in css)
except Exception as e:
    check("CSS", False, str(e))

# 2c. JavaScript
print("\n[2c] JavaScript syntax")
try:
    import subprocess
    for js_file in ["frontend/js/app.js", "frontend/js/sentence.js"]:
        result = subprocess.run(
            ["node", "--check", js_file],
            capture_output=True, text=True, timeout=10
        )
        check(f"{js_file} syntax valid", result.returncode == 0, result.stderr if result.returncode != 0 else "")
except FileNotFoundError:
    check("JavaScript syntax check", False, "node not available, skipping")
except Exception as e:
    check("JavaScript syntax check", False, str(e))

# ---- Phase 3: Containerization ----

print("\n=== PHASE 3: Containerization ===\n")

# 3a. docker-compose.yml
print("[3a] docker-compose.yml")
try:
    import yaml
    with open("docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    check("docker-compose has services", "services" in compose)
    check("Has api service", "api" in compose["services"])
    check("Has frontend service", "frontend" in compose["services"])
    check("api has model volume", "./outputs/models" in str(compose["services"]["api"].get("volumes", [])))
    check("frontend exposes port 80", "80:80" in str(compose["services"]["frontend"].get("ports", [])))
    check("frontend depends on api", "api" in compose["services"]["frontend"].get("depends_on", []))
    check("api has MODEL_PATH env", any("MODEL_PATH" in str(e) for e in compose["services"]["api"].get("environment", [])))
    check("api has MODEL_TYPE env", any("MODEL_TYPE" in str(e) for e in compose["services"]["api"].get("environment", [])))
except ImportError:
    check("yaml available", False, "pyyaml not installed, checking manually")
    with open("docker-compose.yml") as f:
        content = f.read()
    check("docker-compose has services (manual)", "services:" in content)
    check("Has api service (manual)", "api:" in content)
    check("Has frontend service (manual)", "frontend:" in content)
    check("api has model volume (manual)", "./outputs/models" in content)
    check("frontend port 80 (manual)", "80:80" in content)
except Exception as e:
    check("docker-compose.yml", False, str(e))

# 3b. Dockerfiles
print("\n[3b] Dockerfiles")
try:
    with open("api/Dockerfile") as f:
        api_docker = f.read()
    check("api/Dockerfile uses python base", "FROM python:" in api_docker)
    check("api/Dockerfile installs requirements", "requirements.txt" in api_docker)
    check("api/Dockerfile runs uvicorn", "uvicorn" in api_docker)
    check("api/Dockerfile exposes 8000", "8000" in api_docker)

    with open("frontend/Dockerfile") as f:
        fe_docker = f.read()
    check("frontend/Dockerfile uses nginx", "FROM nginx" in fe_docker)
    check("frontend/Dockerfile serves files", "nginx/html" in fe_docker)
except Exception as e:
    check("Dockerfiles", False, str(e))

# 3c. nginx.conf
print("\n[3c] nginx.conf")
try:
    with open("frontend/nginx.conf") as f:
        nginx = f.read()
    check("nginx listens on 80", "listen 80" in nginx)
    check("nginx proxies /api/", "proxy_pass http://api:8000" in nginx)
    check("nginx handles WebSocket", "Upgrade" in nginx)
    check("nginx fallback to index.html", "try_files" in nginx)
except Exception as e:
    check("nginx.conf", False, str(e))

# 3d. .dockerignore
print("\n[3d] .dockerignore")
try:
    with open(".dockerignore") as f:
        ignore = f.read()
    check(".dockerignore excludes .venv", ".venv/" in ignore)
    check(".dockerignore excludes __pycache__", "__pycache__/" in ignore)
    check(".dockerignore excludes datasets", "datasets/" in ignore)
    check(".dockerignore excludes notebooks", "notebooks/" in ignore)
except Exception as e:
    check(".dockerignore", False, str(e))

# ---- Phase 4: Config ----

print("\n=== PHASE 4: Config Externalization ===\n")

try:
    from src.config.settings import (
        MODEL_PATH, MODEL_TYPE,
        PREDICTION_DEFAULTS, BASE_DIR
    )
    check("Settings imports MODEL_PATH", isinstance(MODEL_PATH, str))
    check("Settings imports MODEL_TYPE", isinstance(MODEL_TYPE, str))
    check("MODEL_PATH default", MODEL_PATH == "outputs/models/best_mobilenet_v2.pth")
    check("MODEL_TYPE default", MODEL_TYPE == "mobilenet_v2")
    check("PREDICTION_DEFAULTS has all keys",
          all(k in PREDICTION_DEFAULTS for k in [
              "CONFIDENCE_THRESHOLD", "STABILITY_FRAMES",
              "COOLDOWN_FRAMES", "SMOOTHING_WINDOW", "IMG_SIZE"
          ]))
    check("CONFIDENCE_THRESHOLD is float", isinstance(PREDICTION_DEFAULTS["CONFIDENCE_THRESHOLD"], float))
    check("STABILITY_FRAMES is int", isinstance(PREDICTION_DEFAULTS["STABILITY_FRAMES"], int))

    # Test env var overrides
    os.environ["MODEL_PATH"] = "/custom/model.pth"
    os.environ["MODEL_TYPE"] = "resnet50"
    os.environ["CONFIDENCE_THRESHOLD"] = "0.8"
    os.environ["STABILITY_FRAMES"] = "20"

    # Re-import to pick up env vars
    import importlib
    import src.config.settings as settings_mod
    importlib.reload(settings_mod)

    check("MODEL_PATH env override", settings_mod.MODEL_PATH == "/custom/model.pth")
    check("MODEL_TYPE env override", settings_mod.MODEL_TYPE == "resnet50")
    check("CONFIDENCE_THRESHOLD env override", settings_mod.PREDICTION_DEFAULTS["CONFIDENCE_THRESHOLD"] == 0.8)
    check("STABILITY_FRAMES env override", settings_mod.PREDICTION_DEFAULTS["STABILITY_FRAMES"] == 20)

    # Cleanup
    for key in ["MODEL_PATH", "MODEL_TYPE", "CONFIDENCE_THRESHOLD", "STABILITY_FRAMES"]:
        os.environ.pop(key, None)

except Exception as e:
    import traceback
    check("Config externalization", False, traceback.format_exc())

# ---- Summary ----

print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'='*50}\n")

sys.exit(1 if failed > 0 else 0)
