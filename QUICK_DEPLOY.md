# Quick Deployment Guide

Get the Plant Disease Detector up and running in minutes.

## Prerequisites

- Docker and Docker Compose installed
- Trained model file (`models/best_model.pth`)
- 2GB+ RAM
- (Optional) NVIDIA GPU with CUDA support

---

## Option 1: Docker Compose (Recommended)

**Fastest way to get started!**

```bash
# 1. Ensure you have a trained model
ls models/demo/best_model.pth

# 2. Start all services
docker-compose up -d

# 3. Access the applications
# Web UI: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

That's it! The system is running.

**Stop services:**
```bash
docker-compose down
```

---

## Option 2: Docker (Manual)

### Build Images

```bash
# Build CPU version
docker build --target cpu -t plant-disease-detector:cpu .

# Build API version
docker build --target api -t plant-disease-detector:api .
```

### Run Containers

**Web UI (Streamlit):**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  plant-disease-detector:cpu
```

**REST API (FastAPI):**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/demo/best_model.pth \
  plant-disease-detector:api
```

---

## Option 3: Local Python

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Run Services

**Web UI:**
```bash
python src/cli.py serve --model models/demo/best_model.pth
```

**API Server:**
```bash
uvicorn src.serve.app_api:app --host 0.0.0.0 --port 8000
```

---

## Option 4: Kubernetes

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Container registry

### Deploy

```bash
# 1. Build and push images
docker build --target api -t your-registry/plant-disease-detector:api .
docker push your-registry/plant-disease-detector:api

# 2. Update kubernetes/deployment.yaml with your registry

# 3. Deploy
kubectl apply -f kubernetes/deployment.yaml

# 4. Check status
kubectl get pods -n plant-disease-detector
kubectl get services -n plant-disease-detector
```

---

## Using Make Commands

If you have `make` installed:

```bash
# Install dependencies
make install

# Create dummy dataset
make dataset-dummy

# Train a demo model
make train-demo

# Run local deployment
make deploy-local

# View all commands
make help
```

---

## Quick Test

After deployment, test the API:

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/leaf_image.jpg"
```

---

## Next Steps

1. **Train your own model** (if using dummy model):
   ```bash
   python src/cli.py train --data your_dataset --output models/production
   ```

2. **Configure for production**:
   - Copy `.env.example` to `.env`
   - Set API keys and production settings
   - Review `PRODUCTION_CHECKLIST.md`

3. **Set up monitoring**:
   - View TensorBoard: `tensorboard --logdir models/demo/tensorboard`
   - Configure Prometheus/Grafana
   - Set up logging

4. **Read full documentation**:
   - `DEPLOYMENT.md` - Complete deployment guide
   - `API_REFERENCE.md` - API documentation
   - `README.md` - Full project documentation

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs <container-id>

# Common fix: Check model path
ls models/demo/best_model.pth
```

### Out of memory
```bash
# Reduce batch size or use smaller model
# For Docker, increase memory limit:
docker run --memory="4g" ...
```

### Port already in use
```bash
# Change port:
docker run -p 8502:8501 ...  # Use 8502 instead of 8501
```

### Health check fails
```bash
# Wait 30 seconds after starting
# Run health check script
python health_check.py
```

---

## Getting Help

- Check logs: `docker-compose logs -f`
- Run verification: `python verify_installation.py`
- Review documentation in `docs/`
- Open an issue on GitHub

---

## Production Deployment

For production deployment, see:
- `DEPLOYMENT.md` - Complete deployment guide
- `PRODUCTION_CHECKLIST.md` - Pre-deployment checklist
- `scripts/deploy.sh` or `scripts/deploy.ps1` - Deployment scripts
