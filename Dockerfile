# Multi-stage Dockerfile for Plant Disease Detector
# Supports both CPU and GPU deployments

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .
COPY setup.cfg .

# Create necessary directories
RUN mkdir -p models data logs outputs

# CPU-only stage (default)
FROM base as cpu
CMD ["python", "src/cli.py", "serve", "--host", "0.0.0.0", "--port", "8501"]

# GPU stage (requires NVIDIA Container Toolkit)
FROM base as gpu
# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
CMD ["python", "src/cli.py", "serve", "--host", "0.0.0.0", "--port", "8501"]

# Production API stage with FastAPI
FROM base as api
RUN pip install --no-cache-dir fastapi uvicorn python-multipart
COPY src/serve/app_api.py ./src/serve/
EXPOSE 8000
CMD ["uvicorn", "src.serve.app_api:app", "--host", "0.0.0.0", "--port", "8000"]
