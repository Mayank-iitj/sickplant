# Deployment Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Plant Disease Detector                   │
│                    Production Deployment                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐       ┌──────────────────────┐
│   Users / Clients    │       │   External Systems    │
└──────────┬───────────┘       └──────────┬───────────┘
           │                              │
           │                              │
           ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                           │
│                  (Nginx / Cloud LB)                          │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              │                           │
    ┌─────────▼─────────┐       ┌────────▼────────┐
    │  Streamlit Web UI │       │   FastAPI REST  │
    │   (Port 8501)     │       │   (Port 8000)   │
    │                   │       │                 │
    │  • File Upload    │       │  • /predict     │
    │  • Visualization  │       │  • /batch       │
    │  • Interactive UI │       │  • /explain     │
    └─────────┬─────────┘       └────────┬────────┘
              │                           │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Inference Engine     │
              │                       │
              │  • Model Loading      │
              │  • Preprocessing      │
              │  • Prediction         │
              │  • Grad-CAM          │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   PyTorch Model       │
              │  (ResNet/EfficientNet)│
              │                       │
              │  • GPU Acceleration   │
              │  • Batch Processing   │
              └───────────────────────┘
```

## Deployment Options

### 1. Docker Compose (Local/Small-Scale)

```
┌─────────────────────────────────────────┐
│          Docker Host                     │
│                                          │
│  ┌────────────────┐  ┌────────────────┐ │
│  │  Web Container │  │  API Container │ │
│  │  (Streamlit)   │  │  (FastAPI)     │ │
│  │  Port: 8501    │  │  Port: 8000    │ │
│  └────────┬───────┘  └────────┬───────┘ │
│           │                   │          │
│           └──────┬────────────┘          │
│                  │                       │
│         ┌────────▼────────┐              │
│         │  Shared Volume  │              │
│         │  (Model Files)  │              │
│         └─────────────────┘              │
└─────────────────────────────────────────┘
```

**Command:** `docker-compose up -d`

### 2. Kubernetes (Production/Cloud)

```
┌─────────────────────────────────────────────────────────┐
│                  Kubernetes Cluster                      │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Ingress Controller                  │    │
│  │         (TLS Termination / Routing)              │    │
│  └───────────────────┬─────────────────────────────┘    │
│                      │                                   │
│         ┌────────────┴────────────┐                      │
│         │                         │                      │
│  ┌──────▼──────┐         ┌───────▼──────┐               │
│  │  Web Service│         │  API Service │               │
│  │  (ClusterIP)│         │ (LoadBalancer)│              │
│  └──────┬──────┘         └───────┬──────┘               │
│         │                        │                       │
│  ┌──────▼──────────┐    ┌────────▼──────────┐           │
│  │ Web Deployment  │    │  API Deployment   │           │
│  │  (2-5 Replicas) │    │  (3-10 Replicas)  │           │
│  │                 │    │                   │           │
│  │  ┌───────┐     │    │  ┌───────┐        │           │
│  │  │ Pod 1 │     │    │  │ Pod 1 │        │           │
│  │  └───────┘     │    │  └───────┘        │           │
│  │  ┌───────┐     │    │  ┌───────┐        │           │
│  │  │ Pod 2 │     │    │  │ Pod 2 │        │           │
│  │  └───────┘     │    │  └───────┘        │           │
│  └─────────────────┘    └───────────────────┘           │
│         │                        │                       │
│         └───────────┬────────────┘                       │
│                     │                                    │
│            ┌────────▼────────┐                           │
│            │ Persistent Vol  │                           │
│            │ (Model Storage) │                           │
│            └─────────────────┘                           │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Horizontal Pod Autoscaler                │   │
│  │  • Min: 2 replicas                               │   │
│  │  • Max: 10 replicas                              │   │
│  │  • CPU: 70%, Memory: 80%                         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Command:** `kubectl apply -f kubernetes/deployment.yaml`

### 3. Cloud Services

#### AWS Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         AWS                              │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Route 53 (DNS)                      │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────▼────────────────────────────┐    │
│  │      Application Load Balancer (ALB)            │    │
│  │         • SSL/TLS Termination                    │    │
│  │         • Health Checks                          │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│         ┌─────────────┴─────────────┐                    │
│         │                           │                    │
│  ┌──────▼──────┐           ┌───────▼──────┐             │
│  │  ECS Service│           │ ECS Service  │             │
│  │  (Web UI)   │           │  (API)       │             │
│  │             │           │              │             │
│  │  ┌────────┐ │           │  ┌────────┐  │             │
│  │  │Task 1  │ │           │  │Task 1  │  │             │
│  │  └────────┘ │           │  └────────┘  │             │
│  │  ┌────────┐ │           │  ┌────────┐  │             │
│  │  │Task 2  │ │           │  │Task 2  │  │             │
│  │  └────────┘ │           │  └────────┘  │             │
│  └─────────────┘           └──────────────┘             │
│         │                           │                    │
│         └───────────┬───────────────┘                    │
│                     │                                    │
│            ┌────────▼────────┐                           │
│            │   EFS / S3      │                           │
│            │ (Model Files)   │                           │
│            └─────────────────┘                           │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Monitoring & Logging                            │   │
│  │  • CloudWatch Logs                               │   │
│  │  • CloudWatch Metrics                            │   │
│  │  • X-Ray (Tracing)                               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### GCP Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud                          │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Cloud Load Balancing                     │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────▼────────────────────────────┐    │
│  │             Cloud Run Services                   │    │
│  │                                                   │    │
│  │  ┌──────────────┐      ┌──────────────┐         │    │
│  │  │  Web Service │      │  API Service │         │    │
│  │  │ (Serverless) │      │ (Serverless) │         │    │
│  │  └──────────────┘      └──────────────┘         │    │
│  └─────────────────────────────────────────────────┘    │
│                       │                                  │
│            ┌──────────▼──────────┐                       │
│            │  Cloud Storage      │                       │
│            │  (Model Files)      │                       │
│            └─────────────────────┘                       │
│                                                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Monitoring & Logging                            │   │
│  │  • Cloud Logging                                 │   │
│  │  • Cloud Monitoring                              │   │
│  │  • Cloud Trace                                   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Prediction Request Flow

```
1. Client Request
   │
   ├─> [HTTPS] ──> Load Balancer
   │                    │
   │                    ▼
   │              Route to Service
   │                    │
   │           ┌────────┴─────────┐
   │           │                  │
2. API        ▼                  ▼
   Endpoint   Web UI         REST API
   │           │                  │
   │           ▼                  │
   │      Upload Form            │
   │           │                  │
   │           └──────┬───────────┘
   │                  │
3. Preprocessing     ▼
   │        Validate & Resize Image
   │                  │
   │                  ▼
4. Inference    PyTorch Model
   │           (Forward Pass)
   │                  │
   │                  ▼
5. Post-       Top-K Predictions
   processing        │
   │                 ▼
   │         Format Response
   │                 │
6. Response         ▼
   │           Return JSON
   │                 │
   └────────────────┴──> Client
```

### Explainability Flow

```
1. Request with --explain flag
   │
   ▼
2. Standard Prediction
   │
   ▼
3. Grad-CAM Generation
   │
   ├─> Hook into Layer
   │
   ├─> Compute Gradients
   │
   ├─> Generate Heatmap
   │
   └─> Overlay on Image
   │
   ▼
4. Return Visualization
```

## Scaling Strategy

### Horizontal Scaling

```
Traffic Pattern:
Low  ────────────────────┐
                         │  Scale Up
Medium ──────────┐       │
                 │       │
High ────┐       │       │
         │       │       │
         ▼       ▼       ▼
      ┌────┐ ┌────┐ ┌────┐
      │Pod1│ │Pod2│ │Pod3│ ... (up to 10 pods)
      └────┘ └────┘ └────┘

Metrics:
• CPU > 70% → Add pod
• Memory > 80% → Add pod
• Requests/sec > threshold → Add pod
```

### Vertical Scaling

```
Model Size Impact:

Small Model (ResNet18)
├─ CPU: 1 core
├─ Memory: 2 GB
└─ Latency: ~200ms

Medium Model (EfficientNet-B2)
├─ CPU: 2 cores
├─ Memory: 4 GB
└─ Latency: ~300ms

Large Model (EfficientNet-B4)
├─ CPU: 4 cores or GPU
├─ Memory: 8 GB
└─ Latency: ~500ms (CPU) / ~50ms (GPU)
```

## Monitoring Stack

```
┌─────────────────────────────────────────────────────────┐
│                  Monitoring Architecture                 │
│                                                           │
│  ┌──────────────┐                                        │
│  │ Application  │                                        │
│  │  Metrics     │──────┐                                 │
│  └──────────────┘      │                                 │
│                        │                                 │
│  ┌──────────────┐      ▼                                 │
│  │  Container   │  ┌─────────┐                           │
│  │  Metrics     │─>│Prometheus│                          │
│  └──────────────┘  └────┬────┘                           │
│                         │                                │
│  ┌──────────────┐       │        ┌──────────┐            │
│  │   Logs       │───────┼───────>│ Grafana  │            │
│  └──────────────┘       │        └──────────┘            │
│                         │                                │
│                         ▼                                │
│                   ┌──────────┐                            │
│                   │AlertMgr  │                            │
│                   └────┬─────┘                            │
│                        │                                 │
│                        ▼                                 │
│              ┌──────────────────┐                         │
│              │  Notifications   │                         │
│              │ (Email/Slack/SMS)│                         │
│              └──────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

## Security Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Security Architecture                  │
│                                                           │
│  Layer 1: Network                                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • Firewall Rules                                │    │
│  │  • VPC / Security Groups                         │    │
│  │  • DDoS Protection                               │    │
│  └─────────────────────────────────────────────────┘    │
│                       │                                  │
│  Layer 2: Transport                                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • TLS 1.3                                       │    │
│  │  • Valid Certificates                            │    │
│  │  • Perfect Forward Secrecy                       │    │
│  └─────────────────────────────────────────────────┘    │
│                       │                                  │
│  Layer 3: Application                                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • API Key Authentication                        │    │
│  │  • Rate Limiting                                 │    │
│  │  • Input Validation                              │    │
│  │  • CORS Configuration                            │    │
│  └─────────────────────────────────────────────────┘    │
│                       │                                  │
│  Layer 4: Container                                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • Non-root User                                 │    │
│  │  • Read-only Filesystem                          │    │
│  │  • Security Scanning                             │    │
│  │  • Minimal Base Image                            │    │
│  └─────────────────────────────────────────────────┘    │
│                       │                                  │
│  Layer 5: Data                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  • Encryption at Rest                            │    │
│  │  • Secrets Management                            │    │
│  │  • Access Control                                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Disaster Recovery

```
Primary Region              Backup Region
┌─────────────┐            ┌─────────────┐
│ Active      │            │ Standby     │
│ Deployment  │            │ Deployment  │
│             │            │             │
│ ┌─────────┐ │            │ ┌─────────┐ │
│ │Services │ │   Sync     │ │Services │ │
│ └────┬────┘ │──────────> │ └────┬────┘ │
│      │      │            │      │      │
│ ┌────▼────┐ │            │ ┌────▼────┐ │
│ │Database │ │   Replicate│ │Database │ │
│ └────┬────┘ │──────────> │ └────┬────┘ │
│      │      │            │      │      │
│ ┌────▼────┐ │            │ ┌────▼────┐ │
│ │Storage  │ │   Backup   │ │Storage  │ │
│ └─────────┘ │──────────> │ └─────────┘ │
└─────────────┘            └─────────────┘

RTO: < 30 minutes
RPO: < 1 hour
```

## Cost Optimization

```
Development:
├─ Docker Compose on single server
├─ Cost: ~$50-100/month
└─ Suitable for: Testing, demos

Production (Small):
├─ Kubernetes with 3-5 nodes
├─ Cost: ~$200-500/month
└─ Suitable for: <100 req/min

Production (Medium):
├─ Kubernetes with 5-10 nodes
├─ Auto-scaling enabled
├─ Cost: ~$500-1500/month
└─ Suitable for: 100-1000 req/min

Production (Large):
├─ Multi-region deployment
├─ GPU instances for inference
├─ Cost: ~$2000-5000/month
└─ Suitable for: >1000 req/min
```

---

For detailed deployment instructions, see:
- **QUICK_DEPLOY.md** - Quick start guide
- **DEPLOYMENT.md** - Complete deployment documentation
- **PRODUCTION_CHECKLIST.md** - Pre-deployment checklist
