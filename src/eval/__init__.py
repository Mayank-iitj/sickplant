"""Evaluation module."""

from src.eval.metrics import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    save_evaluation_report,
)

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "plot_confusion_matrix",
    "save_evaluation_report",
]
