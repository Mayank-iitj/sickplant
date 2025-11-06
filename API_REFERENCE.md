# API Reference

Complete REST API reference for the Plant Disease Detector service.

## Base URL

```
http://your-domain.com/api/v1
```

For local development:
```
http://localhost:8000
```

## Authentication

API uses key-based authentication. Include your API key in the request header:

```http
X-API-Key: your-api-key-here
```

## Rate Limiting

- **Default**: 60 requests per minute per IP
- **Authenticated**: 100 requests per minute per API key

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1635724800
```

---

## Endpoints

### Root

#### GET /

Get API information.

**Response:**
```json
{
  "message": "Plant Disease Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### Health Check

#### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "timestamp": "2025-11-04T10:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service is unhealthy

---

### Model Information

#### GET /info

Get model information.

**Response:**
```json
{
  "num_classes": 4,
  "classes": [
    "diseased_mildew",
    "diseased_rust",
    "diseased_spot",
    "healthy"
  ],
  "device": "cuda",
  "model_architecture": "resnet18",
  "image_size": 224
}
```

---

### Single Image Prediction

#### POST /predict

Predict disease from a single image.

**Request:**

Content-Type: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | Image file (JPEG, PNG) |
| top_k | integer | No | Number of top predictions (default: 3, max: 10) |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -H "X-API-Key: your-api-key" \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "diseased_rust",
  "confidence": 0.95,
  "probabilities": {
    "diseased_rust": 0.95,
    "diseased_spot": 0.03,
    "healthy": 0.015,
    "diseased_mildew": 0.005
  },
  "timestamp": "2025-11-04T10:30:00Z",
  "model_info": {
    "device": "cuda",
    "num_classes": 4
  }
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - Invalid file type or parameters
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Prediction failed

---

### Batch Prediction

#### POST /predict/batch

Predict diseases from multiple images.

**Request:**

Content-Type: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| files | file[] | Yes | Multiple image files (max: 50) |
| top_k | integer | No | Number of top predictions per image (default: 3) |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/batch?top_k=3" \
  -H "X-API-Key: your-api-key" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Response:**
```json
[
  {
    "predicted_class": "diseased_rust",
    "confidence": 0.95,
    "probabilities": {
      "diseased_rust": 0.95,
      "diseased_spot": 0.03,
      "healthy": 0.015,
      "diseased_mildew": 0.005
    },
    "timestamp": "2025-11-04T10:30:00Z",
    "model_info": {
      "device": "cuda",
      "num_classes": 4
    }
  },
  {
    "predicted_class": "healthy",
    "confidence": 0.98,
    "probabilities": {
      "healthy": 0.98,
      "diseased_mildew": 0.01,
      "diseased_rust": 0.007,
      "diseased_spot": 0.003
    },
    "timestamp": "2025-11-04T10:30:01Z",
    "model_info": {
      "device": "cuda",
      "num_classes": 4
    }
  }
]
```

**Status Codes:**
- `200 OK` - Predictions successful (partial success returns 200 with fewer results)
- `400 Bad Request` - Too many files or invalid parameters
- `503 Service Unavailable` - Model not loaded

---

### Prediction with Explanation

#### POST /explain

Get prediction with Grad-CAM visualization.

**Request:**

Content-Type: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | Image file (JPEG, PNG) |

**Example:**
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "X-API-Key: your-api-key" \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "diseased_rust",
  "confidence": 0.95,
  "explanation": "Grad-CAM heatmap shows important regions",
  "timestamp": "2025-11-04T10:30:00Z"
}
```

**Response Headers:**
```
X-Prediction-Class: diseased_rust
X-Prediction-Confidence: 0.95
```

**Status Codes:**
- `200 OK` - Explanation generated
- `400 Bad Request` - Invalid file
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Explanation failed

---

### Metrics

#### GET /metrics

Get API metrics (for monitoring).

**Response:**
```json
{
  "total_requests": 0,
  "success_rate": 100.0,
  "average_response_time_ms": 0,
  "uptime_seconds": 0,
  "message": "Metrics endpoint - integrate with monitoring system"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - API key not authorized |
| 404 | Not Found - Endpoint doesn't exist |
| 413 | Payload Too Large - File size exceeds limit |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server-side error |
| 503 | Service Unavailable - Service not ready |

---

## Code Examples

### Python

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
headers = {"X-API-Key": "your-api-key"}

with open("leaf.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, headers=headers, files=files)
    
print(response.json())

# Batch prediction
url = "http://localhost:8000/predict/batch"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
]
response = requests.post(url, headers=headers, files=files)
print(response.json())
```

### JavaScript

```javascript
// Single prediction
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key'
  },
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -F "file=@leaf.jpg"

# With top_k parameter
curl -X POST "http://localhost:8000/predict?top_k=5" \
  -H "X-API-Key: your-api-key" \
  -F "file=@leaf.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "X-API-Key: your-api-key" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

---

## Interactive Documentation

When the API is running, access interactive documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API exploration and testing.

---

## WebSocket Support

WebSocket support for real-time predictions is planned for future releases.

---

## Versioning

API version is included in the base URL. Current version: `v1`

Breaking changes will result in a new version (`v2`, `v3`, etc.).

---

## Support

For API questions or issues:
- GitHub Issues: [repository-url]/issues
- Email: support@yourdomain.com
- Documentation: [docs-url]
