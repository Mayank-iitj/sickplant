#!/bin/bash
# Sample workflow for plant disease detection
# This script demonstrates the complete pipeline from training to inference

echo "=================================="
echo "Plant Disease Detector - Sample Run"
echo "=================================="

# Step 1: Download/prepare dataset
echo ""
echo "Step 1: Preparing dataset..."
python examples/download_dataset.py --dummy

# Step 2: Train model
echo ""
echo "Step 2: Training model..."
python src/cli.py train \
    --data data/dummy_dataset \
    --output models/sample_run \
    --model resnet18 \
    --epochs 5 \
    --batch 8 \
    --lr 0.001 \
    --seed 42

# Step 3: Evaluate model
echo ""
echo "Step 3: Evaluating model..."
python src/cli.py evaluate \
    --model models/sample_run/best_model.pth \
    --data data/dummy_dataset/test \
    --output evaluation/sample_run

# Step 4: Single prediction
echo ""
echo "Step 4: Running inference on sample image..."
# Find a test image
TEST_IMAGE=$(find data/dummy_dataset/test -name "*.jpg" | head -1)

if [ -z "$TEST_IMAGE" ]; then
    echo "No test images found!"
else
    python src/cli.py predict \
        --model models/sample_run/best_model.pth \
        --image "$TEST_IMAGE" \
        --topk 3 \
        --explain \
        --output predictions/
fi

# Step 5: Batch prediction
echo ""
echo "Step 5: Running batch prediction..."
python src/cli.py batch_predict \
    --model models/sample_run/best_model.pth \
    --input-dir data/dummy_dataset/test/healthy \
    --output-dir predictions/batch \
    --format csv

echo ""
echo "=================================="
echo "Sample run completed!"
echo "=================================="
echo ""
echo "Results:"
echo "  - Trained model: models/sample_run/best_model.pth"
echo "  - Evaluation: evaluation/sample_run/"
echo "  - Predictions: predictions/"
echo ""
echo "To launch the web UI:"
echo "  python src/cli.py serve --model models/sample_run/best_model.pth"
