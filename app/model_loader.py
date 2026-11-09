import os
import time
import hashlib
import logging
import threading
import urllib.request
import urllib.error
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"

MODEL_URL = os.getenv("MODEL_URL", "").strip()
MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip().lower()

_model = None
_load_lock = threading.Lock()
_last_load_error = None


class ModelUnavailable(RuntimeError):
    pass


def _is_valid_model_file(path: Path) -> bool:
    """A .pt file is a zip archive. It must be > 1MB and start with PK header."""
    if not path.is_file():
        return False
    if path.stat().st_size < 1_000_000:
        logger.error(f"Model file too small ({path.stat().st_size} bytes) — likely HTML or LFS pointer.")
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    if header[:2] != b"PK":
        logger.error(f"Model file is not a zip archive (header: {header!r}).")
        return False
    return True


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_model():
    """Download model from MODEL_URL to WEIGHTS_PATH safely with verification."""
    if not MODEL_URL:
        raise ModelUnavailable("MODEL_URL environment variable is not set.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = WEIGHTS_PATH.with_suffix(".tmp")

    logger.info(f"Downloading model from {MODEL_URL[:60]}...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AgriHerdAI/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise ModelUnavailable(f"Download failed: HTTP {resp.status}")

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                raise ModelUnavailable(f"Download returned HTML instead of model ({content_type}).")

            total = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            logger.info(f"Downloaded {total} bytes to temp file.")
    except urllib.error.URLError as exc:
        raise ModelUnavailable(f"Network error: {exc}") from exc

    if not _is_valid_model_file(tmp_path):
        tmp_path.unlink(missing_ok=True)
        raise ModelUnavailable("Downloaded file is not a valid PyTorch model.")

    if MODEL_SHA256:
        actual = _compute_sha256(tmp_path)
        logger.info(f"SHA-256: {actual}")
        if actual != MODEL_SHA256:
            tmp_path.unlink(missing_ok=True)
            raise ModelUnavailable(
                f"SHA-256 mismatch. Expected {MODEL_SHA256[:16]}..., got {actual[:16]}..."
            )
        logger.info("SHA-256 verification passed.")
    else:
        logger.warning("MODEL_SHA256 not set — skipping integrity verification.")

    tmp_path.replace(WEIGHTS_PATH)
    logger.info(f"Model installed at {WEIGHTS_PATH}")


def ensure_model_exists():
    global _last_load_error
    if _is_valid_model_file(WEIGHTS_PATH):
        logger.info("Model file already present and valid.")
        return

    try:
        _download_model()
        _last_load_error = None
    except Exception as exc:
        _last_load_error = str(exc)
        raise ModelUnavailable(f"Could not obtain model: {exc}") from exc


def get_model():
    global _model, _last_load_error

    if _model is not None:
        return _model

    with _load_lock:
        if _model is not None:
            return _model

        ensure_model_exists()

        try:
            logger.info("Loading YOLO model into memory...")
            start = time.time()
            _model = YOLO(str(WEIGHTS_PATH))
            logger.info(f"Model loaded in {round(time.time() - start, 2)}s")
            _last_load_error = None
            return _model
        except Exception as exc:
            _last_load_error = str(exc)
            logger.exception("Failed to load YOLO model.")
            if WEIGHTS_PATH.exists():
                WEIGHTS_PATH.unlink()
            raise ModelUnavailable(f"YOLO load failed: {exc}") from exc


def is_model_ready() -> bool:
    return _model is not None


def get_last_error():
    return _last_load_error