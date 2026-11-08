import os
import time
import logging
import threading
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Path to the model inside the container
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best.pt"

_model = None
_load_lock = threading.Lock()

class ModelUnavailable(RuntimeError):
    """Raised when the model file is missing or fails to load."""
    pass

def get_model():
    """
    Lazy loads the YOLO model. Reuses the instance for all subsequent calls.
    """
    global _model

    # Fast path: if already loaded, return immediately (no lock needed)
    if _model is not None:
        return _model

    # Slow path: acquire lock so only ONE thread loads the model
    with _load_lock:
        # Double-check in case another thread loaded it while we waited for the lock
        if _model is not None:
            return _model

        if not WEIGHTS_PATH.is_file():
            logger.error(f"Model file not found at {WEIGHTS_PATH}")
            raise ModelUnavailable("Model file is missing on the server.")

        try:
            logger.info("Loading YOLO model into memory...")
            start_time = time.time()
            
            # Load the model
            _model = YOLO(str(WEIGHTS_PATH))
            
            load_time = round(time.time() - start_time, 2)
            logger.info(f"Model loaded successfully in {load_time} seconds.")
            return _model
        except Exception as e:
            logger.exception("Failed to load YOLO model.")
            raise ModelUnavailable("Failed to load the AI model.") from e