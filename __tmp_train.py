"""Training loop with checkpointing, early stopping, and logging."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.model import PlantDiseaseClassifier, save_checkpoint
from src.utils.io import ensure_dir, save_json

logger = logging.getLogger("plant_disease_detector")


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Use simple comparison operators instead of torch.lt/gt
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """Training orchestrator."""
    
    def __init__(
        self,
        model: PlantDiseaseClassifier,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        log_interval: int = 10,
        use_tensorboard: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device for training
            output_dir: Output directory for checkpoints and logs
            scheduler: Learning rate scheduler
            early_stopping: Early stopping handler
            log_interval: Logging interval in batches
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.log_interval = log_interval
        
        # Create output directory
        ensure_dir(output_dir)
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            tensorboard_dir = os.path.join(output_dir, 'tensorboard')
            ensure_dir(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)
            logger.info(f"TensorBoard logging to {tensorboard_dir}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.start_time = None
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                avg_loss = running_loss / total
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })
                
                # TensorBoard logging
                if self.writer:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                avg_loss = running_loss / total
                acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        num_epochs: int,
        freeze_backbone_epochs: int = 0,
        save_best: bool = True,
        save_last: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            freeze_backbone_epochs: Unfreeze backbone after N epochs
            save_best: Whether to save best model
            save_last: Whether to save last model
            
        Returns:
            Training history dictionary
        """
        logger.info("=" * 80)
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Unfreeze backbone if needed
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
                logger.info("Unfreezing backbone for fine-tuning")
                self.model.unfreeze_backbone()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Log summary
            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/accuracy', train_acc, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/accuracy', val_acc, epoch)
                self.writer.add_scalar('learning_rate', current_lr, epoch)
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                
                best_path = os.path.join(self.output_dir, 'best_model.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    save_path=best_path
                )
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            
            # Save last model
            if save_last:
                last_path = os.path.join(self.output_dir, 'last_model.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    save_path=last_path
                )
            
            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Training complete
        total_time = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best Val Acc: {self.best_val_acc:.2f}%")
        logger.info("=" * 80)
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        save_json(self.history, history_path)
        logger.info(f"Training history saved to {history_path}")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return self.history


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: Optimizer name ('adam' or 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    logger.info(f"Created {optimizer_name.upper()} optimizer (lr={lr}, weight_decay={weight_decay})")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'reduce_on_plateau',
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Scheduler name
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name == 'none' or scheduler_name is None:
        return None
    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 3),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    logger.info(f"Created {scheduler_name} learning rate scheduler")
    
    return scheduler


def create_criterion(
    loss_name: str = 'cross_entropy',
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create loss criterion.
    
    Args:
        loss_name: Loss function name
        class_weights: Class weights for imbalanced data
        label_smoothing: Label smoothing factor
        device: Device to place weights on
        
    Returns:
        Loss function
    """
    if loss_name == 'cross_entropy':
        if class_weights is not None and device is not None:
            class_weights = class_weights.to(device)
        
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    
    logger.info(f"Created {loss_name} loss function")
    if class_weights is not None:
        logger.info(f"Using class weights: {class_weights.tolist()}")
    
    return criterion

