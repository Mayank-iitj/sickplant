"""Command-line interface for plant disease detection."""

import json
import logging
import os
import sys
from pathlib import Path

import click
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import (
    setup_logging,
    set_seed,
    load_config,
    save_config,
    save_json,
    ensure_dir,
    get_device,
    validate_image_file,
)
from src.data.dataset_utils import (
    load_dataset_from_folders,
    load_dataset_from_csv,
    split_dataset,
    PlantDiseaseDataset,
    create_dataloaders,
    compute_class_weights,
)
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.model import create_model, get_supported_backbones
from src.models.train import (
    Trainer,
    EarlyStopping,
    create_optimizer,
    create_scheduler,
    create_criterion,
)
from src.models.inference import PlantDiseasePredictor, save_predictions
from src.eval.metrics import evaluate_model, compute_metrics, save_evaluation_report
from src.explainability.gradcam import GradCAM

logger = logging.getLogger("plant_disease_detector")


@click.group()
def main():
    """Plant Disease Detector CLI."""
    pass


@main.command()
@click.option('--data', required=True, help='Path to dataset directory or CSV file')
@click.option('--output', required=True, help='Output directory for model and logs')
@click.option('--model', default='resnet50', help='Model backbone (default: resnet50)')
@click.option('--epochs', default=30, help='Number of training epochs')
@click.option('--batch', default=32, help='Batch size')
@click.option('--lr', default=0.0001, type=float, help='Learning rate')
@click.option('--freeze-epochs', default=0, help='Epochs to freeze backbone')
@click.option('--early-stopping', default=5, help='Early stopping patience')
@click.option('--config', default=None, help='Path to config YAML file')
@click.option('--gpu/--no-gpu', default=True, help='Use GPU if available')
@click.option('--seed', default=42, help='Random seed')
def train(data, output, model, epochs, batch, lr, freeze_epochs, early_stopping, config, gpu, seed):
    """Train a plant disease detection model."""
    
    # Setup
    ensure_dir(output)
    setup_logging(log_level='INFO', log_file=os.path.join(output, 'train.log'))
    set_seed(seed)
    
    logger.info("=" * 80)
    logger.info("PLANT DISEASE DETECTOR - TRAINING")
    logger.info("=" * 80)
    
    # Load config
    if config:
        cfg = load_config(config)
    else:
        # Default config
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            cfg = load_config(str(config_path))
        else:
            cfg = {}
    
    # Override config with CLI arguments
    cfg['seed'] = seed
    cfg['batch_size'] = batch
    cfg['epochs'] = epochs
    cfg['backbone'] = model
    cfg['freeze_backbone_epochs'] = freeze_epochs
    cfg.setdefault('optimizer', {})['lr'] = lr
    
    # Save config
    save_config(cfg, os.path.join(output, 'config.yaml'))
    
    # Device
    device = get_device(gpu)
    
    # Load dataset
    logger.info(f"Loading dataset from {data}")
    
    if data.endswith('.csv'):
        image_paths, labels, class_names = load_dataset_from_csv(data)
        
        # Split dataset
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
            image_paths, labels,
            train_ratio=cfg.get('data_split', {}).get('train', 0.8),
            val_ratio=cfg.get('data_split', {}).get('val', 0.1),
            test_ratio=cfg.get('data_split', {}).get('test', 0.1),
            seed=seed
        )
    else:
        # Check if pre-split
        train_dir = os.path.join(data, 'train')
        val_dir = os.path.join(data, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            # Pre-split dataset
            logger.info("Detected pre-split dataset")
            train_paths, train_labels, class_names = load_dataset_from_folders(train_dir)
            val_paths, val_labels, _ = load_dataset_from_folders(val_dir)
            test_paths, test_labels = [], []
            
            test_dir = os.path.join(data, 'test')
            if os.path.exists(test_dir):
                test_paths, test_labels, _ = load_dataset_from_folders(test_dir)
        else:
            # Single folder - need to split
            image_paths, labels, class_names = load_dataset_from_folders(data)
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
                image_paths, labels, seed=seed
            )
    
    num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Save class names
    class_names_path = os.path.join(output, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Create datasets
    image_size = tuple(cfg.get('image_size', [224, 224]))
    
    train_transform = get_train_transforms(
        image_size=image_size,
        **cfg.get('augmentation', {})
    )
    val_transform = get_val_transforms(image_size=image_size)
    
    train_dataset = PlantDiseaseDataset(train_paths, train_labels, class_names, train_transform)
    val_dataset = PlantDiseaseDataset(val_paths, val_labels, class_names, val_transform)
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset, val_dataset, None,
        batch_size=batch,
        num_workers=cfg.get('num_workers', 4)
    )
    
    # Create model
    logger.info(f"Creating model: {model}")
    net = create_model(
        num_classes=num_classes,
        backbone=model,
        pretrained=cfg.get('pretrained', True),
        dropout=cfg.get('dropout', 0.3),
        freeze_backbone=(freeze_epochs > 0),
        device=device
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        net,
        optimizer_name=cfg.get('optimizer', {}).get('name', 'adam'),
        lr=lr,
        weight_decay=cfg.get('optimizer', {}).get('weight_decay', 0.0001)
    )
    
    # Create scheduler
    scheduler_cfg = cfg.get('scheduler', {})
    scheduler = create_scheduler(optimizer, **scheduler_cfg)
    
    # Create criterion
    class_weights = None
    if cfg.get('class_weights') == 'auto':
        class_weights = compute_class_weights(train_labels, num_classes)
    
    criterion = create_criterion(
        loss_name=cfg.get('loss', 'cross_entropy'),
        class_weights=class_weights,
        label_smoothing=cfg.get('label_smoothing', 0.0),
        device=device
    )
    
    # Create early stopping
    early_stop = None
    if cfg.get('early_stopping', {}).get('enabled', True):
        early_stop = EarlyStopping(
            patience=early_stopping,
            min_delta=cfg.get('early_stopping', {}).get('min_delta', 0.001),
            mode='min'
        )
    
    # Create trainer
    trainer = Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=output,
        scheduler=scheduler,
        early_stopping=early_stop,
        log_interval=cfg.get('logging', {}).get('log_interval', 10),
        use_tensorboard=cfg.get('logging', {}).get('tensorboard', True)
    )
    
    # Train
    history = trainer.train(
        num_epochs=epochs,
        freeze_backbone_epochs=freeze_epochs
    )
    
    logger.info("Training completed successfully!")


@main.command()
@click.option('--model', required=True, help='Path to trained model file')
@click.option('--data', required=True, help='Path to test dataset directory')
@click.option('--output', default='./evaluation', help='Output directory for results')
@click.option('--batch', default=32, help='Batch size')
def evaluate(model, data, output, batch):
    """Evaluate model on test dataset."""
    
    setup_logging(log_level='INFO')
    ensure_dir(output)
    
    logger.info("=" * 80)
    logger.info("PLANT DISEASE DETECTOR - EVALUATION")
    logger.info("=" * 80)
    
    # Device
    device = get_device()
    
    # Load class names
    model_dir = os.path.dirname(model)
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Classes: {class_names}")
    
    # Load dataset
    logger.info(f"Loading test dataset from {data}")
    test_paths, test_labels, _ = load_dataset_from_folders(data)
    
    # Create dataset and loader
    val_transform = get_val_transforms()
    test_dataset = PlantDiseaseDataset(test_paths, test_labels, class_names, val_transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    logger.info(f"Loading model from {model}")
    checkpoint = torch.load(model, map_location=device)
    
    from src.models.model import PlantDiseaseClassifier
    net = PlantDiseaseClassifier(
        num_classes=len(class_names),
        backbone=checkpoint.get('backbone', 'resnet50'),
        pretrained=False
    )
    
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    
    net.to(device)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred, y_true, y_probs = evaluate_model(net, test_loader, device, class_names)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, class_names)
    
    # Save report
    save_evaluation_report(metrics, y_true, y_pred, class_names, output)
    
    logger.info("Evaluation completed successfully!")


@main.command()
@click.option('--model', required=True, help='Path to trained model file')
@click.option('--image', required=True, help='Path to input image')
@click.option('--topk', default=3, help='Number of top predictions')
@click.option('--threshold', default=0.0, type=float, help='Confidence threshold')
@click.option('--explain', is_flag=True, help='Generate Grad-CAM visualization')
@click.option('--output', default=None, help='Output directory for results')
def predict(model, image, topk, threshold, explain, output):
    """Predict disease class for a single image."""
    
    setup_logging(log_level='INFO')
    
    if output:
        ensure_dir(output)
    
    # Device
    device = get_device()
    
    # Load class names
    model_dir = os.path.dirname(model)
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # Create predictor
    predictor = PlantDiseasePredictor(
        model_path=model,
        class_names=class_names,
        device=device,
        threshold=threshold
    )
    
    # Predict
    result = predictor.predict(image, topk=topk)
    
    # Print result
    print("\n" + "=" * 80)
    print(f"Image: {result['image']}")
    print("=" * 80)
    print(f"Top Prediction: {result['top1']} (confidence: {result['top1_probability']:.4f})")
    print(f"Uncertain: {result['uncertain']}")
    print("\nTop-{} Predictions:".format(topk))
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  {i}. {pred['label']}: {pred['probability']:.4f}")
    print("=" * 80)
    
    # Save result
    if output:
        result_path = os.path.join(output, 'prediction.json')
        save_json(result, result_path)
        logger.info(f"Prediction saved to {result_path}")
    
    # Generate Grad-CAM
    if explain:
        logger.info("Generating Grad-CAM explanation...")
        
        import cv2
        from src.data.augmentations import load_and_preprocess_image
        
        # Load image
        preprocessed, original = load_and_preprocess_image(image)
        input_tensor = preprocessed.unsqueeze(0).to(device)
        
        # Generate Grad-CAM
        gradcam = GradCAM(predictor.model)
        
        overlay_path = os.path.join(output or '.', 'gradcam_overlay.jpg') if output else 'gradcam_overlay.jpg'
        gradcam.save_visualization(original, input_tensor, overlay_path, alpha=0.4)
        
        print(f"\nGrad-CAM visualization saved to: {overlay_path}")


@main.command()
@click.option('--model', required=True, help='Path to trained model file')
@click.option('--input-dir', required=True, help='Input directory with images')
@click.option('--output-dir', required=True, help='Output directory for predictions')
@click.option('--explain', is_flag=True, help='Generate Grad-CAM for all images')
@click.option('--format', default='csv', type=click.Choice(['json', 'csv']), help='Output format')
@click.option('--batch', default=32, help='Batch size')
def batch_predict(model, input_dir, output_dir, explain, format, batch):
    """Process multiple images in batch."""
    
    setup_logging(log_level='INFO')
    ensure_dir(output_dir)
    
    logger.info("=" * 80)
    logger.info("PLANT DISEASE DETECTOR - BATCH PREDICTION")
    logger.info("=" * 80)
    
    # Device
    device = get_device()
    
    # Load class names
    model_dir = os.path.dirname(model)
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # Get image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    image_files = [str(f) for f in image_files if validate_image_file(str(f))]
    
    logger.info(f"Found {len(image_files)} images")
    
    if not image_files:
        logger.error("No valid images found!")
        return
    
    # Create predictor
    predictor = PlantDiseasePredictor(
        model_path=model,
        class_names=class_names,
        device=device
    )
    
    # Batch predict
    results = predictor.predict_batch(image_files, batch_size=batch)
    
    # Save results
    output_file = os.path.join(output_dir, f'predictions.{format}')
    save_predictions(results, output_file, format=format)
    
    # Generate Grad-CAM if requested
    if explain:
        logger.info("Generating Grad-CAM visualizations...")
        
        gradcam_dir = os.path.join(output_dir, 'gradcam')
        ensure_dir(gradcam_dir)
        
        from src.data.augmentations import load_and_preprocess_image
        gradcam = GradCAM(predictor.model)
        
        for img_path in image_files[:10]:  # Limit to first 10 for demo
            try:
                preprocessed, original = load_and_preprocess_image(img_path)
                input_tensor = preprocessed.unsqueeze(0).to(device)
                
                output_path = os.path.join(gradcam_dir, f"gradcam_{Path(img_path).stem}.jpg")
                gradcam.save_visualization(original, input_tensor, output_path)
            except Exception as e:
                logger.error(f"Error generating Grad-CAM for {img_path}: {e}")
    
    logger.info(f"Batch prediction completed! Results saved to {output_file}")


@main.command()
@click.option('--model', required=True, help='Path to trained model file')
@click.option('--port', default=8501, help='Port number')
@click.option('--host', default='localhost', help='Host address')
def serve(model, port, host):
    """Launch Streamlit web UI."""
    
    logger = logging.getLogger("plant_disease_detector")
    setup_logging(log_level='INFO')
    
    logger.info("=" * 80)
    logger.info("PLANT DISEASE DETECTOR - WEB UI")
    logger.info("=" * 80)
    
    # Set environment variables for Streamlit app
    os.environ['MODEL_PATH'] = model
    os.environ['MODEL_DIR'] = os.path.dirname(model)
    
    # Launch Streamlit
    import subprocess
    
    app_path = Path(__file__).parent / 'serve' / 'app_streamlit.py'
    
    cmd = [
        'streamlit', 'run',
        str(app_path),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true'
    ]
    
    logger.info(f"Starting Streamlit app at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Server stopped")


if __name__ == '__main__':
    main()
