# metrics/evaluator.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from .core import compute_classification_metrics
from utils.logger import get_logger
from utils.metrics import plot_confusion,plot_precision_recall
import wandb

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, model, data_loader, class_names=None):
        self.model = model
        self.loader = data_loader
        self.class_names = class_names

    def evaluate(self, verbose=False, save_confusion_path=None, save_pr_path=None,
                 log_to_wandb=False, wandb_prefix=""):

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for batch in self.loader:
                out = self.model(batch)
                prob = torch.exp(out) if out.dim() > 1 else torch.sigmoid(out)
                preds = out.argmax(dim=1)
                y_true.extend(batch.y.tolist())
                y_pred.extend(preds.tolist())
                y_prob.extend(prob.cpu().numpy())

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_prob_np = np.array(y_prob)

        metrics = compute_classification_metrics(y_true_np, y_pred_np, y_prob_np)

        try:
            if y_prob_np.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true_np, y_prob_np[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true_np, y_prob_np, multi_class="ovr")
        except Exception as e:
            logger.warning(f"ROC AUC computation failed: {e}")
            metrics["roc_auc"] = None

        try:
            metrics["log_loss"] = log_loss(y_true_np, y_prob_np)
        except Exception as e:
            logger.warning(f"Log loss computation failed: {e}")
            metrics["log_loss"] = None

        metrics.pop("auroc", None)

        if self.class_names is None:
            self.class_names = [str(i) for i in sorted(set(y_true))]

        if save_confusion_path:
            plot_confusion(y_true, y_pred, class_names=self.class_names, save_path=save_confusion_path)
        if save_pr_path:
            plot_precision_recall(y_true, y_prob_np, save_path=save_pr_path)

        if verbose:
            print("\nClassification Report:\n")
            print(classification_report(y_true, y_pred, target_names=self.class_names))

        if log_to_wandb:
            wandb.log({f"{wandb_prefix}metrics/{k}": v for k, v in metrics.items()})
            if save_confusion_path:
                wandb.log({f"{wandb_prefix}images/confusion_matrix": wandb.Image(save_confusion_path)})
            if save_pr_path:
                wandb.log({f"{wandb_prefix}images/precision_recall": wandb.Image(save_pr_path)})

        return metrics


# --- Utilities ---

def summarize_experiment_results(results_list, output_csv_path):
    df = pd.DataFrame(results_list)
    summary_df = df.agg(['mean', 'std']).T
    summary_df.dropna(subset=["mean"], inplace=True)
    summary_df['summary'] = summary_df.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )
    summary_df.to_csv(output_csv_path)
    logger.info(f"Saved summary to {output_csv_path}")
    return summary_df


def export_latex_table(summary_df, output_path, model_name="Model"):
    latex_str = summary_df[['summary']].to_latex(header=[model_name])
    with open(output_path, 'w') as f:
        f.write(latex_str)
    logger.info(f"Exported LaTeX table to {output_path}")


def export_csv_summary(summary_topo, summary_gat, output_path):
    df_combined = pd.concat([summary_topo[['summary']], summary_gat[['summary']]], axis=1)
    df_combined.columns = ['TopoGAT', 'GAT']
    df_combined.to_csv(output_path)
    logger.info(f"Exported comparison summary to {output_path}")


def pick_representative_seed(results_list, metric='accuracy'):
    df = pd.DataFrame(results_list)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    mean_val = df[metric].mean()
    df['distance'] = np.abs(df[metric] - mean_val)
    idx = df['distance'].idxmin()
    logger.info(f"Representative Seed: {idx} (Closest {metric} to mean {mean_val:.4f})")
    logger.info(f"Representative Metrics: {results_list[idx]}")
    return idx, results_list[idx]


def save_visuals_for_representative_seed(model, loader, class_names, seed_label, output_dir,
                                          log_to_wandb=False, wandb_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    confusion_path = os.path.join(output_dir, f"confusion_{seed_label}.png")
    pr_path = os.path.join(output_dir, f"pr_curve_{seed_label}.png")

    evaluator = ModelEvaluator(model, loader, class_names=class_names)
    evaluator.evaluate(
        verbose=True,
        save_confusion_path=confusion_path,
        save_pr_path=pr_path,
        log_to_wandb=log_to_wandb,
        wandb_prefix=wandb_prefix
    )
