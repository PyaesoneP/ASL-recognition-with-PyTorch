"""FastAPI application for ASL recognition web service."""

import cv2
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from api.models import (
    PredictRequest,
    PredictResponse,
    SentenceUpdateRequest,
    SentenceResponse,
    UpdateRequest,
    UpdateResponse,
    HealthResponse,
)
from api.services.predictor import InferenceService

model_path = os.getenv("MODEL_PATH", "outputs/models/best_mobilenet_v2.onnx")
model_type = os.getenv("MODEL_TYPE", "mobilenet_v2")

service = InferenceService(model_path=model_path, model_type=model_type)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not service._initialized:
            service.initialize()
            print(f"ASL Inference Service initialized: model={model_type}, path={model_path}")
        else:
            print(f"ASL Inference Service already initialized: model={model_type}")
    except Exception as e:
        print(f"WARNING: ASL Inference Service failed to initialize: {e}")
        print("  Endpoints will return error/empty responses until a valid model is available.")
    yield
    print("ASL Inference Service shutting down")


app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition via webcam",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS: allow_credentials must be False when origins is the wildcard, otherwise
# the API would echo any Origin back with credentials allowed. Set ALLOWED_ORIGINS
# (comma-separated) to lock this down to your own domain(s).
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (for single-container deployment on Render)
_frontend_path = Path(__file__).resolve().parents[1] / "frontend"

if _frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_path), html=True), name="frontend")


@app.get("/")
async def serve_index():
    index_file = _frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"detail": "Frontend not found"}


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_type=model_type,
        model_loaded=service._initialized,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        image_bgr = service.decode_base64_image(request.image)
    except Exception:
        return PredictResponse(
            prediction="error",
            confidence=0.0,
            probabilities={},
        )

    label, confidence, probabilities = service.predict(image_bgr, request.session_id)

    if label is None:
        return PredictResponse(
            prediction="error",
            confidence=0.0,
            probabilities={},
        )

    return PredictResponse(
        prediction=label,
        confidence=confidence,
        probabilities=probabilities,
    )


@app.post("/api/sentence/update", response_model=SentenceResponse)
async def update_sentence(request: SentenceUpdateRequest):
    sentence = service.update_sentence(
        session_id=request.session_id,
        action=request.action,
        prediction=request.prediction,
    )
    return SentenceResponse(sentence=sentence, session_id=request.session_id)


@app.get("/api/sentence/{session_id}", response_model=SentenceResponse)
async def get_sentence(session_id: str):
    sentence = service.get_sentence(session_id)
    return SentenceResponse(sentence=sentence, session_id=session_id)


@app.post("/api/update", response_model=UpdateResponse)
async def update(request: UpdateRequest):
    """Documented contract endpoint: POST {class, confidence} → {sentence, added_letter}.

    Architecture-compliant endpoint per architectural_style.md:
    Browser POSTs {class, confidence} to /api/update → {sentence, added_letter}
    """
    sentence = service.update_sentence(
        session_id=request.session_id,
        action="predict",
        prediction=request.class_label,
        confidence=request.confidence,
    )
    # Determine if a letter was actually added (sentence grew)
    added_letter = None
    if request.class_label not in ("del", "space", "nothing"):
        added_letter = request.class_label

    return UpdateResponse(
        sentence=sentence,
        added_letter=added_letter,
        session_id=request.session_id,
    )


@app.websocket("/api/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    # Unique per-connection session so concurrent clients don't share one
    # sentence/smoother. Evicted on disconnect (see finally).
    session_id = uuid.uuid4().hex

    try:
        while True:
            data = await websocket.receive_text()

            if data == "ping":
                await websocket.send_text("pong")
                continue

            import json
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
            action = message.get("action", "predict")

            if action == "clear":
                sentence = service.update_sentence(session_id, "clear")
                await websocket.send_text(json.dumps({
                    "type": "sentence",
                    "sentence": sentence,
                    "prediction": None,
                    "confidence": 0.0,
                }))
            elif action == "space":
                sentence = service.update_sentence(session_id, "space")
                await websocket.send_text(json.dumps({
                    "type": "sentence",
                    "sentence": sentence,
                    "prediction": None,
                    "confidence": 0.0,
                }))
            elif action == "del":
                sentence = service.update_sentence(session_id, "del")
                await websocket.send_text(json.dumps({
                    "type": "sentence",
                    "sentence": sentence,
                    "prediction": None,
                    "confidence": 0.0,
                }))
            elif action == "predict":
                image_b64 = message.get("image")
                if image_b64:
                    try:
                        image_bgr = service.decode_base64_image(image_b64)
                        label, confidence, _ = service.predict(image_bgr, session_id)

                        if label and label != "nothing":
                            service.update_sentence(session_id, "predict", label, confidence)

                        sentence = service.get_sentence(session_id)
                        await websocket.send_text(json.dumps({
                            "type": "prediction",
                            "prediction": label or "nothing",
                            "confidence": confidence or 0.0,
                            "sentence": sentence,
                        }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": str(e),
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "prediction",
                        "prediction": "nothing",
                        "confidence": 0.0,
                        "sentence": service.get_sentence(session_id),
                    }))

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        # A send to an already-closed peer raises a plain RuntimeError (not
        # WebSocketDisconnect). Only attempt to close if the socket is still
        # open, and swallow any close-after-close error so it doesn't surface
        # as an ASGI crash (which would otherwise churn client reconnects and
        # reset per-session sentence/stability state).
        print(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass
    finally:
        service.remove_session(session_id)
