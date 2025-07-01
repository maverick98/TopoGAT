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
    auc
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


def plot_roc_curve(y_true, y_prob, n_classes, class_names=None, save_path=None):
    """
    Plots ROC curve for multi-class classification.

    Args:
        y_true (array-like): True class labels.
        y_prob (array-like): Probabilities for each class.
        n_classes (int): Number of classes.
        class_names (list): Optional names of classes.
        save_path (str): Path to save the ROC curve plot.
    """
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        label = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.2f})")

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
    """
    A reusable evaluator class for computing classification metrics,
    and saving visualizations like confusion matrix and ROC curves.
    """

    def __init__(self, model, data_loader, class_names=None):
        """
        Args:
            model (torch.nn.Module): Trained PyTorch model.
            data_loader (DataLoader): Data loader to evaluate on.
            class_names (list, optional): Class labels for visualization.
        """
        self.model = model
        self.loader = data_loader
        self.class_names = class_names

    def evaluate(self, verbose=False, save_confusion_path=None, save_roc_path=None):
        """
        Runs model inference and computes metrics.

        Args:
            verbose (bool): Whether to print classification report.
            save_confusion_path (str, optional): Path to save confusion matrix.
            save_roc_path (str, optional): Path to save ROC curve.

        Returns:
            dict: Computed classification metrics.
        """
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in self.loader:
                out = self.model(batch)
                probs = torch.exp(out)  # Assuming log_softmax
                preds = probs.argmax(dim=1)
                y_true.extend(batch.y.tolist())
                y_pred.extend(preds.tolist())
                y_prob.extend(probs.tolist())

        y_prob_np = np.array(y_prob)
        metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob_np)

        if save_confusion_path:
            plot_confusion(y_true, y_pred, class_names=self.class_names, save_path=save_confusion_path)

        if save_roc_path:
            n_classes = y_prob_np.shape[1]
            plot_roc_curve(y_true, y_prob_np, n_classes, class_names=self.class_names, save_path=save_roc_path)

        if verbose:
            print("\nClassification Report:\n")
            print(classification_report(y_true, y_pred, target_names=self.class_names))

        return metrics
