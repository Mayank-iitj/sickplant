# Quick Start Guide

## Installation (5 minutes)

### 1. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes depending on your internet connection.

## Quick Test (10 minutes)

### 1. Create Dummy Dataset

```bash
python examples/download_dataset.py --dummy
```

This creates a small dummy dataset at `data/dummy_dataset/` with 4 classes.

### 2. Train a Small Model

```bash
python src/cli.py train \
    --data data/dummy_dataset \
    --output models/test_run \
    --model resnet18 \
    --epochs 3 \
    --batch 8
```

**Expected time:** 2-5 minutes on CPU, <1 minute on GPU

### 3. Test Inference

```bash
# Find a test image (Windows PowerShell)
$img = Get-ChildItem -Path "data\dummy_dataset\test" -Filter "*.jpg" -Recurse | Select-Object -First 1 -ExpandProperty FullName
python src/cli.py predict --model models\test_run\best_model.pth --image $img --topk 3
```

```bash
# Find a test image (Linux/Mac)
img=$(find data/dummy_dataset/test -name "*.jpg" | head -1)
python src/cli.py predict --model models/test_run/best_model.pth --image $img --topk 3
```

### 4. Launch Web UI

```bash
python src/cli.py serve --model models/test_run/best_model.pth
```

Open browser to: http://localhost:8501

## Using Real Data

### Option 1: Download PlantVillage Dataset

1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Setup Kaggle credentials: https://www.kaggle.com/docs/api

3. Download dataset:
   ```bash
   kaggle datasets download -d emmarex/plantdisease
   unzip plantdisease.zip -d data/plant_village
   ```

### Option 2: Use Your Own Images

Organize images in folder-per-class structure:

```
my_dataset/
├── train/
│   ├── healthy/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── diseased/
│       ├── img3.jpg
│       └── img4.jpg
├── val/
│   └── ...
└── test/
    └── ...
```

Then train:

```bash
python src/cli.py train --data my_dataset --output models/my_model
```

## Common Issues

### Import Errors

Make sure you're running from the project root and virtual environment is activated:

```bash
# Check current directory
pwd  # Should be .../plant_disease_detector

# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\Activate.ps1  # Windows
```

### CUDA Out of Memory

Reduce batch size:

```bash
python src/cli.py train --data ... --batch 8  # or even smaller
```

### ModuleNotFoundError

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Experiment with hyperparameters** (learning rate, batch size, epochs)
3. **Try different backbones** (`--model efficientnet_b0`, `mobilenet_v2`, etc.)
4. **Enable Grad-CAM** to visualize model attention (`--explain` flag)
5. **Check training logs** in `models/[run_name]/tensorboard/`

## Getting Help

- Check `python src/cli.py [command] --help` for command details
- See `tests/` directory for usage examples
- Review `config.yaml` for all available settings

## Performance Tips

- **Use GPU** if available (40-100x faster training)
- **Start with small model** (resnet18) for quick iteration
- **Use transfer learning** (enabled by default with `--pretrained`)
- **Monitor with TensorBoard**: `tensorboard --logdir models/[run_name]/tensorboard`

Happy plant disease detecting! 🌱
