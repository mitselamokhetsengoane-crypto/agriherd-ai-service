import logging
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from .ai_service import predict_image
from .security import verify_token
from .model_loader import is_model_ready, get_last_error, WEIGHTS_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgriHerd AI Service", version="1.0.0")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.get("/")
def root():
    return {"service": "AgriHerd AI Disease Scanner", "status": "online"}


@app.get("/health")
def health_check():
    """Liveness — always returns 200 if process is alive. Does NOT require model."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness — fails if model file missing or invalid."""
    if is_model_ready():
        return {"ready": True, "model_loaded": True}

    file_exists = WEIGHTS_PATH.is_file()
    err = get_last_error()
    return {
        "ready": file_exists,
        "model_loaded": False,
        "model_file_present": file_exists,
        "last_error": err,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_token),
):
    logger.info(f"Prediction request: {file.filename}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10MB).")

    result = predict_image(data)
    if "error" in result:
        raise HTTPException(result.get("status_code", 500), result["error"])
    return result