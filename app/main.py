import logging
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from .ai_service import predict_image
from .security import verify_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AgriHerd AI Service", version="1.0.0")

# Max upload size (10 MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

@app.get("/")
def root():
    return {
        "service": "AgriHerd AI Disease Scanner",
        "status": "online",
        "usage": "Send POST requests to /predict with a Bearer token."
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint. 
    DOES NOT load the model. This keeps Cloud Run startup fast and memory low.
    """
    return {"status": "healthy"}

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_token)
):
    """
    Receives an image from the main backend and returns predictions.
    """
    logger.info(f"Received prediction request for file: {file.filename}")
    
    # Read image data
    image_data = await file.read()
    
    if not image_data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image is too large (max 10MB).")

    # Run inference (this will trigger lazy loading on first request)
    result = predict_image(image_data)
    
    if "error" in result:
        raise HTTPException(
            status_code=result.get("status_code", 500), 
            detail=result["error"]
        )
        
    return result