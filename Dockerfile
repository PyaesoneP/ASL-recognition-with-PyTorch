# Production Dockerfile for Render/Railway
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, models, and frontend
COPY src/ /app/src/
COPY api/ /app/api/
COPY outputs/models/ /app/outputs/models/
COPY frontend/ /app/frontend/

# Environment variables
ENV PORT=8000
ENV MODEL_PATH=outputs/models/best_mobilenet_v2.onnx
ENV MODEL_TYPE=mobilenet_v2
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
