"""FastAPI REST API for Plant Disease Detection."""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import io
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from PIL import Image
import numpy as np

from src.models.inference import PlantDiseasePredictor
from src.explainability.gradcam import GradCAM
from src.utils.io import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="REST API for detecting plant diseases from leaf images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[PlantDiseasePredictor] = None
gradcam: Optional[GradCAM] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str = Field(..., description="Predicted disease class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: dict = Field(..., description="Probabilities for all classes")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_info: dict = Field(..., description="Model information")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    image_urls: List[str] = Field(..., description="List of image URLs")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor, gradcam
    
    logger.info("Starting Plant Disease Detection API...")
    
    # Get model path from environment or use default
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.warning("API started without model - predictions will fail")
        return
    
    try:
        # Initialize predictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = PlantDiseasePredictor(model_path=model_path, device=device)
        
        # Initialize Grad-CAM
        gradcam = GradCAM(
            model=predictor.model,
            target_layer=predictor.model.backbone.layer4[-1]
        )
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Using device: {device}")
        logger.info(f"Classes: {predictor.class_names}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
        gradcam = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    device = "unknown"
    if predictor is not None:
        device = str(predictor.device)
    
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        device=device,
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", response_model=dict)
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "num_classes": len(predictor.class_names),
        "classes": predictor.class_names,
        "device": str(predictor.device),
        "model_architecture": predictor.config.get("model", {}).get("backbone", "unknown"),
        "image_size": predictor.config.get("data", {}).get("image_size", 224),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions")
):
    """
    Predict disease from uploaded image.
    
    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return
        
    Returns:
        Prediction results with confidence scores
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction
        result = predictor.predict(image, top_k=top_k)
        
        # Get top prediction
        predicted_class = result["predictions"][0]["class"]
        confidence = result["predictions"][0]["confidence"]
        
        # Format probabilities
        probabilities = {
            pred["class"]: pred["confidence"]
            for pred in result["predictions"]
        }
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now().isoformat(),
            model_info={
                "device": str(predictor.device),
                "num_classes": len(predictor.class_names)
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions")
):
    """
    Predict diseases from multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        top_k: Number of top predictions per image
        
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images per batch request"
        )
    
    results = []
    
    for file in files:
        try:
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Get prediction
            result = predictor.predict(image, top_k=top_k)
            
            predicted_class = result["predictions"][0]["class"]
            confidence = result["predictions"][0]["confidence"]
            
            probabilities = {
                pred["class"]: pred["confidence"]
                for pred in result["predictions"]
            }
            
            results.append(PredictionResponse(
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                timestamp=datetime.now().isoformat(),
                model_info={
                    "device": str(predictor.device),
                    "num_classes": len(predictor.class_names)
                }
            ))
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            # Continue with other images
            continue
    
    return results


@app.post("/explain")
async def explain_prediction(
    file: UploadFile = File(..., description="Image file to explain")
):
    """
    Get prediction with Grad-CAM visualization.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction with explanation heatmap
    """
    if predictor is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction
        result = predictor.predict(image, top_k=3)
        predicted_class = result["predictions"][0]["class"]
        confidence = result["predictions"][0]["confidence"]
        
        # Get predicted class index
        class_idx = predictor.class_names.index(predicted_class)
        
        # Generate Grad-CAM
        image_np = np.array(image)
        cam_result = gradcam.generate_cam(image_np, class_idx=class_idx)
        
        # Convert overlay to bytes
        overlay_image = Image.fromarray(cam_result["overlay"])
        img_byte_arr = io.BytesIO()
        overlay_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return JSONResponse(
            content={
                "predicted_class": predicted_class,
                "confidence": confidence,
                "explanation": "Grad-CAM heatmap shows important regions",
                "timestamp": datetime.now().isoformat()
            },
            headers={
                "X-Prediction-Class": predicted_class,
                "X-Prediction-Confidence": str(confidence)
            }
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/metrics", response_model=dict)
async def get_metrics():
    """Get API metrics (placeholder for production monitoring)."""
    return {
        "total_requests": 0,
        "success_rate": 100.0,
        "average_response_time_ms": 0,
        "uptime_seconds": 0,
        "message": "Metrics endpoint - integrate with monitoring system"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
