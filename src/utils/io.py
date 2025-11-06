"""Utility functions for I/O operations, logging, and reproducibility."""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("plant_disease_detector")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        JSON data as dictionary
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: Dict[str, Any], output_path: str, indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data dictionary
        output_path: Path to save JSON file
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    os.makedirs(directory, exist_ok=True)


def get_device(gpu: bool = True) -> torch.device:
    """
    Get the appropriate device (CPU or GPU).
    
    Args:
        gpu: Whether to use GPU if available
        
    Returns:
        torch.device object
    """
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger("plant_disease_detector")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger = logging.getLogger("plant_disease_detector")
        logger.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_class_names(class_names_path: str) -> list:
    """
    Load class names from a text file (one class per line).
    
    Args:
        class_names_path: Path to class names file
        
    Returns:
        List of class names
    """
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def save_class_names(class_names: list, output_path: str) -> None:
    """
    Save class names to a text file.
    
    Args:
        class_names: List of class names
        output_path: Path to save class names file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")


def validate_image_file(image_path: str) -> bool:
    """
    Check if a file is a valid image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid image, False otherwise
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    path = Path(image_path)
    
    if not path.exists():
        return False
    
    if path.suffix.lower() not in valid_extensions:
        return False
    
    # Try to open the image to verify it's not corrupted
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_model_size(model_path: str) -> str:
    """
    Get human-readable model file size.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Formatted file size string
    """
    size_bytes = os.path.getsize(model_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"
