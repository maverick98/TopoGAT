import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize


def compute_classification_metrics(y_true, y_pred, y_prob=None, average="macro"):
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


def plot_confusion(y_true, y_pred, class_names=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
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

def plot_roc_curve(y_true, y_prob, n_classes, class_names=None, save_path=None):
    plt.figure(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            label = class_names[i] if class_names else f"Class {i}"
            plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


class ModelEvaluator:
    def __init__(self, model, data_loader, class_names=None, use_log_softmax=True):
        self.model = model
        self.loader = data_loader
        self.class_names = class_names
        self.use_log_softmax = use_log_softmax

    def evaluate(self, verbose=False, save_confusion_path=None, save_roc_path=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(device)
                out = self.model(batch)
                if self.use_log_softmax:
                    probs = torch.exp(out)
                else:
                    probs = torch.softmax(out, dim=-1)
                preds = probs.argmax(dim=1)
                y_true.extend(batch.y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        y_prob_np = np.array(y_prob)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob_np)

        if save_confusion_path:
            plot_confusion(y_true, y_pred, class_names=self.class_names, save_path=save_confusion_path)

        if save_roc_path:
            n_classes = y_prob_np.shape[1]
            plot_roc_curve(y_true, y_prob_np, n_classes, class_names=self.class_names, save_path=save_roc_path)

        if verbose:
            try:
                print("\nClassification Report:\n")
                print(classification_report(
                    y_true, y_pred,
                    target_names=self.class_names or [str(i) for i in range(y_prob_np.shape[1])]
                ))
            except Exception as e:
                print("Error printing classification report:", e)

        return metrics
