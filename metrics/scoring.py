# metrics/scoring.py
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def compute_classification_metrics(y_true, y_pred, y_prob=None, average="macro"):
    """
    Computes standard classification metrics.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for ROC-AUC.
        average (str): Averaging method for multi-class metrics.

    Returns:
        dict: Dictionary with accuracy, precision, recall, F1, and optionally ROC-AUC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            metrics["roc_auc"] = None
    return metrics
