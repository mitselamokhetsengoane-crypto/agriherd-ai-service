FROM python:3.11-slim

# Hugging Face requires a user with UID 1000
RUN useradd -m -u 1000 user
USER user

WORKDIR /home/user/app

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and weights
COPY --chown=user app/ ./app/
COPY --chown=user weights/ ./weights/

# Hugging Face exposes port 7860 by default
ENV PORT=7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]