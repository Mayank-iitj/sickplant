# Sample workflow for plant disease detection (PowerShell version)
# This script demonstrates the complete pipeline from training to inference

Write-Host "==================================" -ForegroundColor Green
Write-Host "Plant Disease Detector - Sample Run" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Step 1: Download/prepare dataset
Write-Host ""
Write-Host "Step 1: Preparing dataset..." -ForegroundColor Yellow
python examples/download_dataset.py --dummy

# Step 2: Train model
Write-Host ""
Write-Host "Step 2: Training model..." -ForegroundColor Yellow
python src/cli.py train `
    --data data/dummy_dataset `
    --output models/sample_run `
    --model resnet18 `
    --epochs 5 `
    --batch 8 `
    --lr 0.001 `
    --seed 42

# Step 3: Evaluate model
Write-Host ""
Write-Host "Step 3: Evaluating model..." -ForegroundColor Yellow
python src/cli.py evaluate `
    --model models/sample_run/best_model.pth `
    --data data/dummy_dataset/test `
    --output evaluation/sample_run

# Step 4: Single prediction
Write-Host ""
Write-Host "Step 4: Running inference on sample image..." -ForegroundColor Yellow

# Find a test image
$TEST_IMAGE = Get-ChildItem -Path "data/dummy_dataset/test" -Filter "*.jpg" -Recurse | Select-Object -First 1 -ExpandProperty FullName

if ($TEST_IMAGE) {
    python src/cli.py predict `
        --model models/sample_run/best_model.pth `
        --image $TEST_IMAGE `
        --topk 3 `
        --explain `
        --output predictions/
} else {
    Write-Host "No test images found!" -ForegroundColor Red
}

# Step 5: Batch prediction
Write-Host ""
Write-Host "Step 5: Running batch prediction..." -ForegroundColor Yellow
python src/cli.py batch_predict `
    --model models/sample_run/best_model.pth `
    --input-dir data/dummy_dataset/test/healthy `
    --output-dir predictions/batch `
    --format csv

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "Sample run completed!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:"
Write-Host "  - Trained model: models/sample_run/best_model.pth"
Write-Host "  - Evaluation: evaluation/sample_run/"
Write-Host "  - Predictions: predictions/"
Write-Host ""
Write-Host "To launch the web UI:" -ForegroundColor Cyan
Write-Host "  python src/cli.py serve --model models/sample_run/best_model.pth"
