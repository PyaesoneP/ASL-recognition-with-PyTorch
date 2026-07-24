"""FastAPI application for ASL recognition web service."""

import cv2
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    session_id = "default"

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
                            service.update_sentence(session_id, "predict", label)

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
        print(f"WebSocket error: {e}")
        await websocket.close()
