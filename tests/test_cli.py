"""Tests for CLI commands."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_import():
    """Test that CLI module can be imported."""
    try:
        from src import cli
        assert cli is not None
    except ImportError as e:
        pytest.fail(f"Failed to import CLI: {e}")


def test_supported_backbones():
    """Test that supported backbones list is available."""
    from src.models.model import get_supported_backbones
    
    backbones = get_supported_backbones()
    
    assert isinstance(backbones, list)
    assert len(backbones) > 0
    assert 'resnet50' in backbones
    assert 'efficientnet_b0' in backbones


def test_config_loading():
    """Test configuration loading."""
    from src.utils.io import load_config, save_config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test config
        config = {
            'seed': 42,
            'batch_size': 32,
            'epochs': 10
        }
        
        config_path = os.path.join(tmpdir, 'test_config.yaml')
        save_config(config, config_path)
        
        # Load config
        loaded_config = load_config(config_path)
        
        assert loaded_config['seed'] == 42
        assert loaded_config['batch_size'] == 32
        assert loaded_config['epochs'] == 10


def test_logging_setup():
    """Test logging setup."""
    from src.utils.io import setup_logging
    
    logger = setup_logging(log_level='INFO')
    
    assert logger is not None
    assert logger.name == 'plant_disease_detector'


def test_seed_setting():
    """Test seed setting for reproducibility."""
    from src.utils.io import set_seed
    import random
    import numpy as np
    import torch
    
    # Set seed
    set_seed(42)
    
    # Generate random numbers
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()
    
    # Reset seed
    set_seed(42)
    
    # Generate again - should be same
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()
    
    assert r1 == r2
    assert n1 == n2
    assert t1 == t2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
