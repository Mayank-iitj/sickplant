# Deployment Guide

Complete guide for deploying the Plant Disease Detection system in production environments.

## Table of Contents
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Logging](#monitoring--logging)
- [Security](#security)

---

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+ (optional)
- Trained model file (`best_model.pth`)

### Quick Start with Docker

#### 1. Build the Image

```bash
# CPU version
docker build --target cpu -t plant-disease-detector:cpu .

# GPU version (requires NVIDIA Container Toolkit)
docker build --target gpu -t plant-disease-detector:gpu .

# API version
docker build --target api -t plant-disease-detector:api .
```

#### 2. Run Container

```bash
# Run Streamlit web UI (CPU)
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  plant-disease-detector:cpu

# Run FastAPI REST API
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/best_model.pth \
  plant-disease-detector:api

# Run with GPU support
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/models:/app/models:ro \
  plant-disease-detector:gpu
```

#### 3. Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

### Docker Compose Deployment

```bash
# Start all services (CPU)
docker-compose up -d

# Start with GPU support
docker-compose --profile gpu up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services exposed:
- Web UI: http://localhost:8501
- REST API: http://localhost:8000
- GPU Web UI: http://localhost:8502 (with `--profile gpu`)

### Docker Configuration

Environment variables:
```bash
MODEL_PATH=/app/models/best_model.pth
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0  # For GPU deployments
PYTHONUNBUFFERED=1
```

---

## Kubernetes Deployment

### Prerequisites
- Kubernetes 1.20+
- kubectl configured
- Container registry access
- Persistent volume for model storage

### Setup

#### 1. Push Images to Registry

```bash
# Tag images
docker tag plant-disease-detector:cpu your-registry/plant-disease-detector:latest
docker tag plant-disease-detector:api your-registry/plant-disease-detector:api-latest

# Push to registry
docker push your-registry/plant-disease-detector:latest
docker push your-registry/plant-disease-detector:api-latest
```

#### 2. Update Kubernetes Manifests

Edit `kubernetes/deployment.yaml`:
```yaml
image: your-registry/plant-disease-detector:api-latest
```

#### 3. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -n plant-disease-detector
kubectl get services -n plant-disease-detector

# View logs
kubectl logs -f deployment/plant-disease-api -n plant-disease-detector
```

#### 4. Upload Model to Persistent Volume

```bash
# Copy model to PVC
kubectl cp models/best_model.pth \
  plant-disease-detector/$(kubectl get pod -n plant-disease-detector \
  -l app=plant-disease-api -o jsonpath='{.items[0].metadata.name}'):/app/models/
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment plant-disease-api --replicas=5 -n plant-disease-detector

# Check HPA status
kubectl get hpa -n plant-disease-detector
```

### Access Services

```bash
# Get service URLs
kubectl get services -n plant-disease-detector

# Port forward for testing
kubectl port-forward -n plant-disease-detector service/plant-disease-api-service 8000:80
```

---

## Cloud Deployment

### AWS Deployment

#### ECS (Elastic Container Service)

1. **Push to ECR**:
```bash
aws ecr create-repository --repository-name plant-disease-detector
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag plant-disease-detector:api <account-id>.dkr.ecr.us-east-1.amazonaws.com/plant-disease-detector:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/plant-disease-detector:latest
```

2. **Create ECS Task Definition**
3. **Create ECS Service with Load Balancer**
4. **Store model in S3 and mount via EFS**

#### Lambda (Serverless)

For inference only:
```python
# Create Lambda deployment package with model
# Use API Gateway for HTTP endpoint
# Store model in S3, load on cold start
```

### Google Cloud Platform

#### Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/plant-disease-detector
gcloud run deploy plant-disease-detector \
  --image gcr.io/PROJECT-ID/plant-disease-detector \
  --platform managed \
  --memory 4Gi \
  --cpu 2
```

#### GKE (Google Kubernetes Engine)

Use the Kubernetes manifests with GKE-specific configurations.

### Azure

#### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name plant-disease-detector \
  --image your-registry/plant-disease-detector:api-latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --dns-name-label plant-disease-api
```

#### Azure Kubernetes Service (AKS)

Use the Kubernetes manifests with AKS.

---

## Performance Optimization

### Model Optimization

#### 1. ONNX Export

```python
# Export to ONNX format
python src/cli.py export --model models/best_model.pth --format onnx --output models/model.onnx
```

#### 2. TorchScript

```python
import torch
from src.models.model import PlantDiseaseClassifier

model = PlantDiseaseClassifier(num_classes=4, backbone='resnet18')
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save('models/model_scripted.pt')
```

#### 3. Quantization

```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Server Optimization

#### Gunicorn Configuration

```bash
# Run API with multiple workers
gunicorn src.serve.app_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --max-requests 1000 \
  --max-requests-jitter 100
```

#### NGINX Reverse Proxy

```nginx
upstream api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

### Caching

```python
# Add caching to API
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="plant-disease-cache")
```

---

## Monitoring & Logging

### Application Monitoring

#### Prometheus Metrics

```python
# Add to FastAPI app
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

#### Health Checks

```python
# Kubernetes liveness/readiness probes
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready"}
```

### Logging

#### Structured Logging

```python
import structlog

logger = structlog.get_logger()
logger.info("prediction_made", 
            class_name=predicted_class,
            confidence=confidence,
            duration_ms=duration)
```

#### Log Aggregation

**ELK Stack**:
```yaml
# Filebeat configuration
filebeat.inputs:
- type: log
  paths:
    - /app/logs/*.log
  json.keys_under_root: true
  
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

**CloudWatch** (AWS):
```python
# Configure CloudWatch logging
import watchtower
import logging

logger = logging.getLogger()
logger.addHandler(watchtower.CloudWatchLogHandler())
```

### Alerting

**Prometheus AlertManager**:
```yaml
groups:
- name: plant_disease_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    annotations:
      summary: "High error rate detected"
```

---

## Security

### API Security

#### 1. API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Security(verify_api_key)])
async def predict(...):
    ...
```

#### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    ...
```

#### 3. HTTPS/TLS

```bash
# Use Let's Encrypt certificates
certbot --nginx -d api.yourdomain.com
```

### Container Security

#### 1. Non-root User

```dockerfile
# Add to Dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

#### 2. Security Scanning

```bash
# Scan for vulnerabilities
trivy image plant-disease-detector:api

# Scan during build
docker scan plant-disease-detector:api
```

### Network Security

```yaml
# Kubernetes Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: plant-disease-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
```

---

## Load Testing

### Apache Bench

```bash
ab -n 1000 -c 10 -p image.jpg -T "image/jpeg" http://localhost:8000/predict
```

### Locust

```python
from locust import HttpUser, task, between

class PlantDiseaseUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        with open("test_image.jpg", "rb") as f:
            self.client.post("/predict", files={"file": f})
```

Run: `locust -f locustfile.py --host http://localhost:8000`

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Use smaller model
   - Enable model quantization

2. **Slow Inference**:
   - Use GPU
   - Enable TorchScript
   - Reduce image size

3. **Container Won't Start**:
   - Check logs: `docker logs <container-id>`
   - Verify model path
   - Check port conflicts

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG plant-disease-detector:api
```

---

## Production Checklist

- [ ] Model trained and validated
- [ ] Docker images built and tested
- [ ] Environment variables configured
- [ ] Secrets management setup
- [ ] SSL/TLS certificates configured
- [ ] Monitoring and logging enabled
- [ ] Backup strategy defined
- [ ] Rate limiting configured
- [ ] API authentication enabled
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Disaster recovery plan
- [ ] CI/CD pipeline configured
- [ ] Security scan passed

---

## Support

For deployment issues:
- Check logs first
- Review configuration
- Consult documentation
- Open GitHub issue

For production support, contact: [your-email@domain.com]
