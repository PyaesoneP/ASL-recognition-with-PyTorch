"""Pydantic request/response schemas for the ASL recognition API."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG image (cropped hand ROI)")
    session_id: str = Field(default="default", description="Session identifier for sentence tracking")


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Predicted ASL class label")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    probabilities: Optional[dict] = Field(None, description="Full class probability distribution")


class SentenceUpdateRequest(BaseModel):
    session_id: str = Field(default="default", description="Session identifier for sentence tracking")
    action: str = Field(default="predict", description="Action: predict, space, del, clear")
    prediction: Optional[str] = Field(None, description="Prediction letter (for 'predict' action)")


class SentenceResponse(BaseModel):
    sentence: str = Field(..., description="Current accumulated sentence")
    session_id: str = Field(..., description="Session identifier")
    added_letter: Optional[str] = Field(None, description="Letter that was just added (if any)")


class UpdateRequest(BaseModel):
    """Request for the /api/update endpoint (documented contract)."""
    class_label: str = Field(..., description="Predicted class label (e.g. 'A')")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    session_id: str = Field(default="default", description="Session identifier")


class UpdateResponse(BaseModel):
    """Response for the /api/update endpoint (documented contract)."""
    sentence: str = Field(..., description="Current accumulated sentence")
    added_letter: Optional[str] = Field(None, description="Letter that was just added (if any)")
    session_id: str = Field(..., description="Session identifier")


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool
