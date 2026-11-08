FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV/YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install globally (so uvicorn is in PATH)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and weights
COPY app/ /app/app/
COPY weights/ /app/weights/

# Render automatically sets $PORT, but we default to 8000 just in case
ENV PORT=8000

# Run the FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]