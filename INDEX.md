# 🌱 Plant Disease Detector - Start Here!

Welcome to the Plant Disease Detector! This file will guide you to the right documentation.

## 🚀 Quick Navigation

### New Users - Start Here!
1. **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in 10 minutes
2. **[README.md](README.md)** - Complete documentation and usage guide
3. **Run verification**: `python verify_installation.py`

### Developers
1. **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
2. **[FILE_MANIFEST.md](FILE_MANIFEST.md)** - Complete project structure
3. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical details and features

### Quick Commands

```bash
# Install
pip install -r requirements.txt

# Create test dataset
python examples/download_dataset.py --dummy

# Train
python src/cli.py train --data data/dummy_dataset --output models/test --epochs 3

# Predict
python src/cli.py predict --model models/test/best_model.pth --image path/to/image.jpg --explain

# Launch Web UI
python src/cli.py serve --model models/test/best_model.pth
```

## 📖 Documentation Index

| Document | Purpose | Who Should Read |
|----------|---------|-----------------|
| [QUICKSTART.md](QUICKSTART.md) | 10-minute setup guide | Everyone |
| [README.md](README.md) | Complete user documentation | Users & Developers |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Feature list & tech details | Developers & Evaluators |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines | Contributors |
| [FILE_MANIFEST.md](FILE_MANIFEST.md) | Complete file structure | Developers |

## 🎯 Common Tasks

### Training a Model
```bash
python src/cli.py train \
    --data your_dataset/ \
    --output models/my_model \
    --model resnet50 \
    --epochs 30 \
    --batch 32
```
**📘 Read:** [README.md#Training](README.md) for all options

### Making Predictions
```bash
# Single image
python src/cli.py predict --model models/my_model/best_model.pth --image leaf.jpg --topk 3 --explain

# Batch processing
python src/cli.py batch_predict --model models/my_model/best_model.pth --input-dir images/ --output-dir results/
```
**📘 Read:** [README.md#Prediction](README.md) for details

### Evaluating a Model
```bash
python src/cli.py evaluate --model models/my_model/best_model.pth --data test_dataset/ --output eval_results/
```
**📘 Read:** [README.md#Evaluation](README.md)

### Using the Web UI
```bash
python src/cli.py serve --model models/my_model/best_model.pth --port 8501
```
Then open: http://localhost:8501

**📘 Read:** [README.md#Web-UI](README.md)

## 🔧 Troubleshooting

**Installation Issues?**
- Run: `python verify_installation.py`
- Check: [QUICKSTART.md#Common-Issues](QUICKSTART.md)

**Training Issues?**
- See: [README.md#Troubleshooting](README.md)
- Reduce batch size: `--batch 8`
- Try smaller model: `--model resnet18`

**Import Errors?**
- Activate virtual environment
- Reinstall: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.9+)

## 📊 Project Structure Overview

```
plant_disease_detector/
├── src/            # Source code
│   ├── cli.py      # Command-line interface
│   ├── data/       # Data loading & augmentation
│   ├── models/     # Model architecture & training
│   ├── eval/       # Evaluation & metrics
│   ├── explainability/  # Grad-CAM
│   └── serve/      # Web UI
├── tests/          # Unit tests
├── examples/       # Example scripts
└── docs/           # Documentation (you are here!)
```

**📘 Full structure:** [FILE_MANIFEST.md](FILE_MANIFEST.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

**📘 Read:** [CONTRIBUTING.md#Testing](CONTRIBUTING.md)

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `python examples/download_dataset.py --dummy`
3. Train small model: 3 epochs on dummy data
4. Try web UI
5. Make predictions

### Intermediate
1. Read full [README.md](README.md)
2. Download real dataset (PlantVillage)
3. Train production model with various backbones
4. Experiment with hyperparameters
5. Analyze results with evaluation metrics

### Advanced
1. Read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Read [CONTRIBUTING.md](CONTRIBUTING.md)
3. Explore source code in `src/`
4. Add new features
5. Contribute back!

## 💡 Key Features

- ✅ **5 CLI commands** for complete workflow
- ✅ **15+ model backbones** (ResNet, EfficientNet, MobileNet, etc.)
- ✅ **Grad-CAM explainability** visualization
- ✅ **Web UI** for easy interaction
- ✅ **Comprehensive metrics** (accuracy, F1, confusion matrix)
- ✅ **Production-ready** with tests & docs

## 🆘 Getting Help

1. Check documentation (start with [QUICKSTART.md](QUICKSTART.md))
2. Run `python src/cli.py [command] --help`
3. Review examples in `examples/`
4. Check [README.md#Troubleshooting](README.md)
5. Open an issue on GitHub

## 📞 Contact & Contributing

Want to contribute? Read [CONTRIBUTING.md](CONTRIBUTING.md)

Found a bug? Open an issue!

Have questions? Check documentation first, then ask!

## ⭐ Quick Reference Card

```bash
# Help
python src/cli.py --help
python src/cli.py train --help

# Train
python src/cli.py train --data DATA --output OUTPUT [options]

# Evaluate  
python src/cli.py evaluate --model MODEL --data DATA --output OUTPUT

# Predict
python src/cli.py predict --model MODEL --image IMAGE [--explain]

# Batch
python src/cli.py batch_predict --model MODEL --input-dir DIR --output-dir DIR

# Serve
python src/cli.py serve --model MODEL [--port PORT]
```

---

**Ready to start? Go to [QUICKSTART.md](QUICKSTART.md)!** 🚀
