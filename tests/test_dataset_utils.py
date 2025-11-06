"""Tests for dataset utilities."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_utils import (
    load_dataset_from_folders,
    split_dataset,
    PlantDiseaseDataset,
    compute_class_weights,
)
from src.data.augmentations import get_train_transforms, get_val_transforms


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create class folders
        classes = ['healthy', 'diseased']
        
        for class_name in classes:
            class_dir = os.path.join(tmpdir, class_name)
            os.makedirs(class_dir)
            
            # Create dummy images
            for i in range(5):
                img_path = os.path.join(class_dir, f'img_{i}.jpg')
                # Create a simple dummy image file
                from PIL import Image
                img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
                img.save(img_path)
        
        yield tmpdir, classes


def test_load_dataset_from_folders(dummy_dataset):
    """Test loading dataset from folder structure."""
    data_dir, expected_classes = dummy_dataset
    
    image_paths, labels, class_names = load_dataset_from_folders(data_dir)
    
    # Check number of images
    assert len(image_paths) == 10  # 5 images per class * 2 classes
    assert len(labels) == 10
    
    # Check class names
    assert set(class_names) == set(expected_classes)
    
    # Check labels are valid indices
    assert all(0 <= label < len(class_names) for label in labels)


def test_split_dataset(dummy_dataset):
    """Test dataset splitting."""
    data_dir, _ = dummy_dataset
    
    image_paths, labels, class_names = load_dataset_from_folders(data_dir)
    
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths, labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )
    
    # Check split sizes
    assert len(train_paths) == len(train_labels)
    assert len(val_paths) == len(val_labels)
    assert len(test_paths) == len(test_labels)
    
    # Check total equals original
    total = len(train_paths) + len(val_paths) + len(test_paths)
    assert total == len(image_paths)
    
    # Check no overlap
    all_paths = set(train_paths + val_paths + test_paths)
    assert len(all_paths) == total


def test_plant_disease_dataset(dummy_dataset):
    """Test PlantDiseaseDataset."""
    data_dir, _ = dummy_dataset
    
    image_paths, labels, class_names = load_dataset_from_folders(data_dir)
    transform = get_val_transforms(image_size=(224, 224))
    
    dataset = PlantDiseaseDataset(
        image_paths=image_paths,
        labels=labels,
        class_names=class_names,
        transform=transform,
        validate_images=True
    )
    
    # Check dataset size
    assert len(dataset) == len(image_paths)
    
    # Check data retrieval
    image, label = dataset[0]
    
    # Check image shape (C, H, W)
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)
    
    # Check label
    assert isinstance(label, int)
    assert 0 <= label < len(class_names)


def test_compute_class_weights():
    """Test class weight computation."""
    # Imbalanced labels
    labels = [0, 0, 0, 0, 1, 2]  # Class 0 has 4 samples, classes 1 and 2 have 1 each
    num_classes = 3
    
    weights = compute_class_weights(labels, num_classes)
    
    # Check shape
    assert weights.shape == (num_classes,)
    
    # Check that minority classes have higher weights
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_transforms():
    """Test augmentation transforms."""
    from PIL import Image
    
    # Create dummy image
    img = Image.new('RGB', (300, 300), color='red')
    img_array = np.array(img)
    
    # Test training transforms
    train_transform = get_train_transforms(image_size=(224, 224))
    augmented = train_transform(image=img_array)
    assert augmented['image'].shape == (3, 224, 224)
    
    # Test validation transforms
    val_transform = get_val_transforms(image_size=(224, 224))
    augmented = val_transform(image=img_array)
    assert augmented['image'].shape == (3, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
