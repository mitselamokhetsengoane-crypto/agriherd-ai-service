import logging
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
from .model_loader import get_model, ModelUnavailable

logger = logging.getLogger(__name__)

MAX_SIDE_PX = 1280
MIN_CONFIDENCE = 0.25
CONFIDENT_AT = 0.50

def predict_image(image_bytes: bytes) -> dict:
    """
    Accepts raw image bytes, validates, preprocesses, and runs inference.
    """
    try:
        model = get_model()
    except ModelUnavailable as exc:
        return {"error": str(exc), "status_code": 503}

    try:
        # 1. Validate and preprocess the image
        with Image.open(BytesIO(image_bytes)) as probe:
            probe.verify()
            
        image = Image.open(BytesIO(image_bytes))
        image.load()
        # Handle EXIF rotation (phone photos) and convert to RGB
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Resize if too large
        image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
        
    except UnidentifiedImageError:
        return {"error": "Unable to read image. Upload a valid JPEG, PNG, or WEBP.", "status_code": 400}
    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}", "status_code": 400}

    try:
        # 2. Run inference
        results = model.predict(image, imgsz=640, conf=MIN_CONFIDENCE, verbose=False)
        res = results[0]
        names = model.names
        detections = []
        
        boxes = getattr(res, "boxes", None)
        probs = getattr(res, "probs", None)

        if boxes is not None and len(boxes):
            for box in boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "class_name": names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                })
        elif probs is not None:
            class_id = int(probs.top1)
            detections.append({
                "class_name": names[class_id],
                "confidence": float(probs.top1conf),
                "bbox": None
            })

        # Sort by confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        top = detections[0] if detections else None
        
        return {
            "detections": detections,
            "status": "completed" if top else "no_detection",
            "low_confidence": bool(top and top["confidence"] < CONFIDENT_AT),
            "message": None if top else "No known condition recognized."
        }
        
    except Exception as e:
        logger.exception("AI model failed to process image")
        return {"error": "The AI model could not process this image.", "status_code": 500}