"""Evaluation metrics and reporting."""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

logger = logging.getLogger("plant_disease_detector")


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    class_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device for evaluation
        class_names: List of class names
        
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Metrics dictionary
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and micro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Per-class metrics dictionary
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx])
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'micro_precision': float(precision_micro),
        'micro_recall': float(recall_micro),
        'micro_f1': float(f1_micro),
        'per_class': per_class_metrics
    }
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize by true class
        figsize: Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(
    metrics: Dict,
    save_path: str,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    per_class = metrics['per_class']
    class_names = list(per_class.keys())
    
    precision = [per_class[c]['precision'] for c in class_names]
    recall = [per_class[c]['recall'] for c in class_names]
    f1 = [per_class[c]['f1'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-class metrics plot saved to {save_path}")


def save_evaluation_report(
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """
    Save comprehensive evaluation report.
    
    Args:
        metrics: Metrics dictionary
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")
    
    # Save confusion matrices
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path, normalize=True)
    
    cm_raw_path = os.path.join(output_dir, 'confusion_matrix_raw.png')
    plot_confusion_matrix(y_true, y_pred, class_names, cm_raw_path, normalize=False)
    
    # Save per-class metrics plot
    metrics_plot_path = os.path.join(output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics, metrics_plot_path)
    
    # Save per-class metrics CSV
    per_class_df = pd.DataFrame(metrics['per_class']).T
    per_class_csv = os.path.join(output_dir, 'per_class_metrics.csv')
    per_class_df.to_csv(per_class_csv)
    logger.info(f"Per-class metrics CSV saved to {per_class_csv}")
    
    logger.info(f"Evaluation report saved to {output_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print("\nPer-Class Metrics:")
    print(per_class_df.to_string())
    print("=" * 80)
