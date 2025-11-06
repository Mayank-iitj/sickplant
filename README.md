# Plant Disease Detector

A production-ready, end-to-end Python tool for detecting plant diseases from leaf images using deep learning.

## Features

- 🔬 **Multi-class disease detection** with confidence scores
- 🎯 **Transfer learning** using pre-trained backbones (ResNet50, EfficientNet)
- 📊 **Explainability** with Grad-CAM visualization
- 🖼️ **Flexible input** - single images, batches, or webcam
- 💻 **CLI** for training, evaluation, and inference
- 🌐 **Web UI** (Streamlit) for interactive predictions
- 📈 **Comprehensive evaluation** - confusion matrix, metrics, reports
- ✅ **Unit tested** and production-ready

## Quick Start

### Installation

```bash
# Clone the repository
cd plant_disease_detector

# Create virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Sample Dataset

```bash
# Download and prepare the PlantVillage dataset (subset)
python examples/download_dataset.py
```

This will download a sample dataset with the following structure:
```
data/plant_village/
├── train/
│   ├── healthy/
│   ├── powdery_mildew/
│   ├── leaf_spot/
│   └── rust/
├── val/
└── test/
```

### Training a Model

```bash
# Train with default settings (ResNet50, 30 epochs)
python src/cli.py train --data data/plant_village --output ./models/run1

# Train with custom hyperparameters
python src/cli.py train \
    --data data/plant_village \
    --model efficientnet_b0 \
    --epochs 50 \
    --batch 16 \
    --lr 0.0001 \
    --output ./models/run2

# See all options
python src/cli.py train --help
```

### Single Image Prediction

```bash
# Predict with top-3 results
python src/cli.py predict \
    --model ./models/run1/best_model.pth \
    --image examples/sample_leaf.jpg \
    --topk 3

# With Grad-CAM explainability
python src/cli.py predict \
    --model ./models/run1/best_model.pth \
    --image examples/sample_leaf.jpg \
    --explain \
    --output results/
```

### Batch Prediction

```bash
# Process entire folder
python src/cli.py batch_predict \
    --model ./models/run1/best_model.pth \
    --input_dir images/ \
    --output_dir predictions/ \
    --explain
```

### Model Evaluation

```bash
# Evaluate on test set
python src/cli.py evaluate \
    --model ./models/run1/best_model.pth \
    --data data/plant_village/test \
    --output ./evaluation/
```

### Web UI

```bash
# Launch Streamlit app
python src/cli.py serve \
    --model ./models/run1/best_model.pth \
    --port 8501
```

Then open your browser to `http://localhost:8501`

## Project Structure

```
plant_disease_detector/
├── README.md
├── requirements.txt
├── setup.cfg
├── config.yaml              # Default training configuration
├── src/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_utils.py    # Dataset loading and splitting
│   │   └── augmentations.py    # Image augmentation pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py            # Model architecture
│   │   ├── train.py            # Training loop
│   │   └── inference.py        # Inference engine
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py          # Grad-CAM visualization
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py          # Evaluation metrics
│   ├── serve/
│   │   ├── __init__.py
│   │   └── app_streamlit.py    # Web UI
│   └── utils/
│       ├── __init__.py
│       └── io.py               # I/O utilities
├── tests/
│   ├── __init__.py
│   ├── test_dataset_utils.py
│   ├── test_inference.py
│   └── test_cli.py
└── examples/
    ├── download_dataset.py     # Download sample data
    ├── sample_run.sh          # Example workflow
    └── sample_images/         # Sample test images
```

## Configuration

Default settings are in `config.yaml`:

```yaml
seed: 42
image_size: [224, 224]
batch_size: 32
epochs: 30
backbone: resnet50
pretrained: true
optimizer:
  name: adam
  lr: 0.0001
scheduler:
  name: reduce_on_plateau
  patience: 3
  factor: 0.1
loss: cross_entropy
augmentation:
  rotate: true
  flip: true
  brightness: 0.2
  contrast: 0.2
  blur: true
early_stopping: 5
```

You can override these via CLI arguments or create custom config files.

## Dataset Format

### Folder-per-Class (Recommended)

```
dataset/
├── healthy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── powdery_mildew/
│   ├── img1.jpg
│   └── ...
└── rust/
    └── ...
```

### CSV Manifest (Alternative)

Create a CSV file with `image_path,label`:

```csv
image_path,label
images/001.jpg,healthy
images/002.jpg,powdery_mildew
images/003.jpg,rust
```

Then use: `python src/cli.py train --data dataset.csv --data_format csv`

## Model Architectures

Supported backbones (all with ImageNet pre-training):
- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`
- `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

## CLI Reference

### Train
```bash
python src/cli.py train [OPTIONS]

Options:
  --data PATH              Path to dataset directory or CSV file [required]
  --output PATH            Output directory for model and logs [required]
  --model TEXT             Model backbone (default: resnet50)
  --epochs INT             Number of training epochs (default: 30)
  --batch INT              Batch size (default: 32)
  --lr FLOAT              Learning rate (default: 0.0001)
  --freeze-epochs INT      Epochs to freeze backbone (default: 0)
  --early-stopping INT     Early stopping patience (default: 5)
  --config PATH           Path to config YAML file
  --gpu / --no-gpu        Use GPU if available (default: auto-detect)
```

### Predict
```bash
python src/cli.py predict [OPTIONS]

Options:
  --model PATH            Path to trained model file [required]
  --image PATH            Path to input image [required]
  --topk INT              Number of top predictions (default: 3)
  --threshold FLOAT       Confidence threshold (default: 0.0)
  --explain               Generate Grad-CAM visualization
  --output PATH           Output directory for results
```

### Batch Predict
```bash
python src/cli.py batch_predict [OPTIONS]

Options:
  --model PATH            Path to trained model file [required]
  --input_dir PATH        Input directory with images [required]
  --output_dir PATH       Output directory for predictions [required]
  --explain               Generate Grad-CAM for all images
  --format TEXT           Output format: json or csv (default: csv)
```

### Evaluate
```bash
python src/cli.py evaluate [OPTIONS]

Options:
  --model PATH            Path to trained model file [required]
  --data PATH             Path to test dataset directory [required]
  --output PATH           Output directory for evaluation results
  --batch INT             Batch size (default: 32)
```

### Serve
```bash
python src/cli.py serve [OPTIONS]

Options:
  --model PATH            Path to trained model file [required]
  --port INT              Port number (default: 8501)
  --host TEXT             Host address (default: localhost)
```

## Output Format

### Prediction JSON
```json
{
  "image": "sample_leaf.jpg",
  "predictions": [
    {"label": "powdery_mildew", "probability": 0.87, "confidence": 0.87},
    {"label": "healthy", "probability": 0.08, "confidence": 0.08},
    {"label": "rust", "probability": 0.03, "confidence": 0.03}
  ],
  "top1": "powdery_mildew",
  "top1_probability": 0.87,
  "uncertain": false,
  "timestamp": "2025-11-04T10:30:00"
}
```

### Evaluation Metrics
```json
{
  "accuracy": 0.9234,
  "macro_f1": 0.9123,
  "micro_f1": 0.9234,
  "per_class": {
    "healthy": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "support": 120},
    "powdery_mildew": {"precision": 0.89, "recall": 0.91, "f1": 0.90, "support": 98},
    "rust": {"precision": 0.92, "recall": 0.90, "f1": 0.91, "support": 105}
  }
}
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_dataset_utils.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Performance Benchmarks

On a sample PlantVillage dataset (4 classes, 8,000 images):
- **Training time**: ~15 min (GPU), ~2 hours (CPU)
- **Inference time**: ~50ms per image (GPU), ~200ms (CPU)
- **Expected accuracy**: >90% validation accuracy with ResNet50
- **Model size**: ~90MB (ResNet50), ~20MB (MobileNetV2)

## Advanced Features

### Transfer Learning with Frozen Backbone
```bash
# Freeze backbone for first 10 epochs, then fine-tune
python src/cli.py train \
    --data data/plant_village \
    --model resnet50 \
    --freeze-epochs 10 \
    --epochs 30 \
    --output ./models/frozen_start
```

### Class Imbalance Handling
```bash
# Automatic class weight computation
python src/cli.py train \
    --data data/plant_village \
    --class-weights auto \
    --output ./models/balanced
```

### Model Export for Deployment
```bash
# Export to ONNX format
python examples/export_model.py \
    --model ./models/run1/best_model.pth \
    --output ./models/run1/model.onnx
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch 16` or `--batch 8`
- Use smaller model: `--model mobilenet_v2`
- Reduce image size in config.yaml

### Low Accuracy
- Train longer: `--epochs 50`
- Try different backbone: `--model efficientnet_b0`
- Check data quality and class balance
- Enable stronger augmentation in config

### Slow Training
- Enable GPU: ensure CUDA is installed
- Increase batch size if memory allows: `--batch 64`
- Use smaller model for prototyping: `--model mobilenet_v2`

## Requirements

- Python 3.9+
- 4GB+ RAM (8GB+ recommended)
- GPU optional but recommended for training
- ~500MB disk space for dependencies
- ~100MB per trained model

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{plant_disease_detector,
  title={Plant Disease Detector: An End-to-End Deep Learning Tool},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/plant_disease_detector}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Acknowledgments

- PlantVillage dataset
- PyTorch/TorchVision team
- Grad-CAM implementation inspired by jacobgil/pytorch-grad-cam

## Contact

For issues and questions, please open a GitHub issue or contact [your-email@example.com]
