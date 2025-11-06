"""Tests for inference module."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model import PlantDiseaseClassifier, save_checkpoint
from src.models.inference import PlantDiseasePredictor, save_predictions


@pytest.fixture
def dummy_model():
    """Create a dummy model and save it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model
        model = PlantDiseaseClassifier(
            num_classes=3,
            backbone='resnet18',
            pretrained=False
        )
        
        # Save model
        model_path = os.path.join(tmpdir, 'model.pth')
        optimizer = torch.optim.Adam(model.parameters())
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            val_accuracy=0.8,
            save_path=model_path
        )
        
        # Save class names
        class_names = ['healthy', 'rust', 'blight']
        class_names_path = os.path.join(tmpdir, 'class_names.txt')
        with open(class_names_path, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        yield model_path, class_names, tmpdir


@pytest.fixture
def dummy_image():
    """Create a dummy image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from PIL import Image
        
        img = Image.new('RGB', (224, 224), color='green')
        img_path = os.path.join(tmpdir, 'test_image.jpg')
        img.save(img_path)
        
        yield img_path


def test_predictor_initialization(dummy_model):
    """Test predictor initialization."""
    model_path, class_names, _ = dummy_model
    
    predictor = PlantDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        device=torch.device('cpu')
    )
    
    assert predictor.num_classes == 3
    assert predictor.class_names == class_names
    assert predictor.model is not None


def test_single_prediction(dummy_model, dummy_image):
    """Test single image prediction."""
    model_path, class_names, _ = dummy_model
    
    predictor = PlantDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        device=torch.device('cpu')
    )
    
    result = predictor.predict(dummy_image, topk=3)
    
    # Check result structure
    assert 'image' in result
    assert 'predictions' in result
    assert 'top1' in result
    assert 'top1_probability' in result
    assert 'uncertain' in result
    
    # Check predictions
    assert len(result['predictions']) == 3
    assert result['top1'] in class_names
    assert 0 <= result['top1_probability'] <= 1


def test_batch_prediction(dummy_model, dummy_image):
    """Test batch prediction."""
    model_path, class_names, _ = dummy_model
    
    predictor = PlantDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        device=torch.device('cpu')
    )
    
    # Create list of images
    image_paths = [dummy_image] * 3
    
    results = predictor.predict_batch(image_paths, batch_size=2, topk=3)
    
    # Check results
    assert len(results) == 3
    
    for result in results:
        assert 'predictions' in result
        assert 'top1' in result


def test_save_predictions():
    """Test saving predictions."""
    predictions = [
        {
            'image': 'img1.jpg',
            'top1': 'healthy',
            'top1_probability': 0.95,
            'predictions': [
                {'label': 'healthy', 'probability': 0.95},
                {'label': 'rust', 'probability': 0.03},
                {'label': 'blight', 'probability': 0.02}
            ],
            'uncertain': False
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test JSON format
        json_path = os.path.join(tmpdir, 'predictions.json')
        save_predictions(predictions, json_path, format='json')
        assert os.path.exists(json_path)
        
        # Test CSV format
        csv_path = os.path.join(tmpdir, 'predictions.csv')
        save_predictions(predictions, csv_path, format='csv')
        assert os.path.exists(csv_path)


def test_threshold_handling(dummy_model, dummy_image):
    """Test uncertainty threshold handling."""
    model_path, class_names, _ = dummy_model
    
    # High threshold - should mark as uncertain
    predictor = PlantDiseasePredictor(
        model_path=model_path,
        class_names=class_names,
        device=torch.device('cpu'),
        threshold=0.99  # Very high threshold
    )
    
    result = predictor.predict(dummy_image)
    
    # Should be marked as uncertain since threshold is very high
    assert 'uncertain' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
