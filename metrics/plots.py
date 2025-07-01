# metrics/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

def plot_confusion(y_true, y_pred, class_names=None, save_path=None):
    """
    Plots and optionally saves a confusion matrix.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): Class labels for the axis.
        save_path (str, optional): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_precision_recall(y_true, y_score, save_path=None):
    """
    Plots and optionally saves the Precision-Recall curve.

    Args:
        y_true (array-like): Ground truth labels.
        y_score (array-like): Probability scores.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(6, 5))
    if y_score.ndim == 2 and y_score.shape[1] > 1:
        for i in range(y_score.shape[1]):
            precision, recall, _ = precision_recall_curve((np.array(y_true) == i).astype(int), y_score[:, i])
            ap = average_precision_score((np.array(y_true) == i).astype(int), y_score[:, i])
            plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(recall, precision, label=f"AP={ap:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
