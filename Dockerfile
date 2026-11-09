FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + wget to download the model
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY app/ /app/app/

# Create weights directory and download the model directly from GitHub Raw
RUN mkdir -p /app/weights
# REPLACE THE URL BELOW WITH YOUR ACTUAL RAW GITHUB URL
RUN wget -O /app/weights/best.pt "https://github.com/mitselamokhetsengoane-crypto/agriherd-ai-service/raw/refs/heads/main/weights/best.pt"

ENV PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]