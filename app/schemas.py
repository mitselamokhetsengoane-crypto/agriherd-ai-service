from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: Optional[List[float]] = None

class PredictionResponse(BaseModel):
    detections: List[Detection]
    status: str  # "completed" or "no_detection"
    low_confidence: bool
    message: Optional[str] = None