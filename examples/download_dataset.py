"""Download and prepare sample PlantVillage dataset."""

import os
import sys
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import setup_logging, ensure_dir

logger = setup_logging(log_level='INFO')


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_plantvillage_dataset():
    """
    Download a sample subset of the PlantVillage dataset.
    
    Note: This downloads a subset for demonstration. For the full dataset,
    visit: https://www.kaggle.com/datasets/emmarex/plantdisease
    """
    logger.info("=" * 80)
    logger.info("DOWNLOADING SAMPLE PLANT DISEASE DATASET")
    logger.info("=" * 80)
    
    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data'
    ensure_dir(str(data_dir))
    
    output_dir = data_dir / 'plant_village'
    
    # Check if already downloaded
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"Dataset already exists at {output_dir}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            logger.info("Using existing dataset")
            return
    
    ensure_dir(str(output_dir))
    
    # Create sample dataset structure manually
    # In a real scenario, you would download from a URL
    logger.info("Creating sample dataset structure...")
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        ensure_dir(str(split_dir))
        
        # Create class folders
        classes = [
            'healthy',
            'powdery_mildew',
            'leaf_spot',
            'rust'
        ]
        
        for class_name in classes:
            class_dir = split_dir / class_name
            ensure_dir(str(class_dir))
            
            # Create placeholder README
            readme_path = class_dir / 'README.txt'
            with open(readme_path, 'w') as f:
                f.write(f"""
{class_name.replace('_', ' ').title()} - {split.upper()} SET

This directory should contain images of plant leaves with {class_name.replace('_', ' ')}.

To use your own dataset:
1. Remove this README.txt file
2. Add your image files (.jpg, .jpeg, .png) to this directory
3. Ensure images are clear and well-lit
4. Recommended image size: at least 224x224 pixels

For a public dataset, download the PlantVillage dataset from:
- Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset

Or use the PlantDoc dataset:
- https://github.com/pratikkayal/PlantDoc-Dataset
""")
    
    logger.info(f"Sample dataset structure created at: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Download a real plant disease dataset:")
    logger.info("   - PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease")
    logger.info("   - PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset")
    logger.info("2. Extract and organize images into the created folders")
    logger.info("3. Or use your own images following the folder-per-class structure")
    logger.info("\nAlternatively, you can download datasets using kaggle API:")
    logger.info("   kaggle datasets download -d emmarex/plantdisease")
    
    # Create sample download script
    download_script = data_dir / 'download_kaggle_dataset.sh'
    with open(download_script, 'w') as f:
        f.write("""#!/bin/bash
# Download PlantVillage dataset from Kaggle
# Requires: pip install kaggle
# Setup: https://www.kaggle.com/docs/api

echo "Downloading PlantVillage dataset from Kaggle..."
kaggle datasets download -d emmarex/plantdisease -p ./data

echo "Extracting dataset..."
unzip ./data/plantdisease.zip -d ./data/plant_village

echo "Organizing dataset..."
# Add your organization logic here if needed

echo "Done! Dataset ready at ./data/plant_village"
""")
    
    logger.info(f"\nKaggle download script created at: {download_script}")
    logger.info("Make it executable with: chmod +x {}".format(download_script))


def create_dummy_images():
    """Create a small dummy dataset for quick testing."""
    logger.info("Creating dummy dataset for testing...")
    
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    data_dir = Path(__file__).parent.parent / 'data' / 'dummy_dataset'
    
    classes = ['healthy', 'diseased_rust', 'diseased_spot', 'diseased_mildew']
    
    for split in ['train', 'val', 'test']:
        for class_name in classes:
            class_dir = data_dir / split / class_name
            ensure_dir(str(class_dir))
            
            # Number of images per split
            num_images = {'train': 20, 'val': 5, 'test': 5}[split]
            
            for i in range(num_images):
                # Create colored image based on class
                if 'healthy' in class_name:
                    color = (50, 200, 50)  # Green
                elif 'rust' in class_name:
                    color = (200, 100, 50)  # Orange-brown
                elif 'spot' in class_name:
                    color = (100, 150, 50)  # Yellow-green
                else:
                    color = (200, 200, 200)  # Gray
                
                # Add some variation
                color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in color)
                
                img = Image.new('RGB', (224, 224), color=color)
                draw = ImageDraw.Draw(img)
                
                # Add some random shapes to simulate leaf patterns
                for _ in range(10):
                    x = random.randint(0, 200)
                    y = random.randint(0, 200)
                    r = random.randint(5, 20)
                    draw.ellipse([x, y, x+r, y+r], fill=(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    ))
                
                # Save
                img_path = class_dir / f'{class_name}_{split}_{i:03d}.jpg'
                img.save(img_path)
    
    logger.info(f"Dummy dataset created at: {data_dir}")
    logger.info("This dataset can be used for quick testing and development.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download plant disease dataset')
    parser.add_argument('--dummy', action='store_true', help='Create dummy dataset for testing')
    args = parser.parse_args()
    
    if args.dummy:
        create_dummy_images()
    else:
        download_plantvillage_dataset()
        
        response = input("\nCreate a small dummy dataset for testing? (y/n): ")
        if response.lower() == 'y':
            create_dummy_images()
