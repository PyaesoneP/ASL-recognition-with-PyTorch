FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying src/ into /app/src to maintain the package structure expected by imports
COPY src/ /app/src/
COPY api/ /app/api/

# Default environment variables for local/container testing
# In production, these can be overridden via container environment settings
ENV MODEL_PATH=outputs/models/best_mobilenet_v2.onnx
ENV MODEL_TYPE=mobilenet_v2
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
