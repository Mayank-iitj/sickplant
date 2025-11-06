"""Verify installation and setup of Plant Disease Detector."""

import sys
from pathlib import Path

print("=" * 80)
print("PLANT DISEASE DETECTOR - INSTALLATION VERIFICATION")
print("=" * 80)
print()

# Check Python version
print("1. Checking Python version...")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 9:
    print(f"   ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    print(f"   ✗ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    print("   Error: Python 3.9+ required")
    sys.exit(1)

# Check imports
print("\n2. Checking dependencies...")
required_packages = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("PIL", "Pillow"),
    ("cv2", "OpenCV"),
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("albumentations", "Albumentations"),
    ("streamlit", "Streamlit"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
    ("click", "Click"),
    ("timm", "timm"),
]

missing_packages = []
for module_name, package_name in required_packages:
    try:
        __import__(module_name)
        print(f"   ✓ {package_name}")
    except ImportError:
        print(f"   ✗ {package_name}")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n   Error: Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
    sys.exit(1)

# Check CUDA availability
print("\n3. Checking GPU support...")
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"     CUDA version: {torch.version.cuda}")
else:
    print("   ℹ CUDA not available (will use CPU)")

# Check project structure
print("\n4. Checking project structure...")
base_dir = Path(__file__).parent.parent
required_dirs = [
    "src/data",
    "src/models",
    "src/eval",
    "src/explainability",
    "src/serve",
    "src/utils",
    "tests",
    "examples",
]

required_files = [
    "src/cli.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
]

for dir_path in required_dirs:
    full_path = base_dir / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}/")
    else:
        print(f"   ✗ {dir_path}/")

for file_path in required_files:
    full_path = base_dir / file_path
    if full_path.exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path}")

# Test imports from src
print("\n5. Testing module imports...")
sys.path.insert(0, str(base_dir))

try:
    from src.utils.io import setup_logging, set_seed
    print("   ✓ src.utils.io")
except ImportError as e:
    print(f"   ✗ src.utils.io: {e}")

try:
    from src.data.dataset_utils import PlantDiseaseDataset
    print("   ✓ src.data.dataset_utils")
except ImportError as e:
    print(f"   ✗ src.data.dataset_utils: {e}")

try:
    from src.models.model import PlantDiseaseClassifier
    print("   ✓ src.models.model")
except ImportError as e:
    print(f"   ✗ src.models.model: {e}")

try:
    from src.models.inference import PlantDiseasePredictor
    print("   ✓ src.models.inference")
except ImportError as e:
    print(f"   ✗ src.models.inference: {e}")

try:
    from src.explainability.gradcam import GradCAM
    print("   ✓ src.explainability.gradcam")
except ImportError as e:
    print(f"   ✗ src.explainability.gradcam: {e}")

# Test model creation
print("\n6. Testing model creation...")
try:
    from src.models.model import create_model
    model = create_model(num_classes=4, backbone='resnet18', pretrained=False)
    print(f"   ✓ Model created successfully")
    print(f"     Parameters: {model.count_parameters():,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("1. Create a dataset:")
print("   python examples/download_dataset.py --dummy")
print()
print("2. Train a test model:")
print("   python src/cli.py train --data data/dummy_dataset --output models/test --epochs 3")
print()
print("3. Read the documentation:")
print("   - README.md for full documentation")
print("   - QUICKSTART.md for quick start guide")
print()
print("For help: python src/cli.py --help")
print("=" * 80)
