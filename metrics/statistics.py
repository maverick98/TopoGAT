import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def paired_ttest_with_effect_size(results_a, results_b, metric_names=None):
    """
    Computes paired t-tests and Cohen's d between two result sets.

    Args:
        results_a (list of dicts): First model's evaluation results.
        results_b (list of dicts): Second model's evaluation results.
        metric_names (list): Optional list of metric names to compare.

    Returns:
        dict: metric -> (t-stat, p-value, cohen's d)
    """
    if metric_names is None:
        metric_names = list(results_a[0].keys())

    stats = {}
    for metric in metric_names:
        a = np.array([res[metric] for res in results_a])
        b = np.array([res[metric] for res in results_b])
        t_stat, p_val = ttest_rel(a, b)
        cohen_d = (a - b).mean() / a.std()
        stats[metric] = {"t": t_stat, "p": p_val, "d": cohen_d}
    return stats


def plot_comparison_boxplots(results_a, results_b, label_a, label_b, metric_names, title, save_path):
    """
    Plots boxplots for comparing two model results.

    Args:
        results_a (list of dicts): First model's metric results.
        results_b (list of dicts): Second model's metric results.
        label_a (str): Label for model A.
        label_b (str): Label for model B.
        metric_names (list): List of metrics to plot.
        title (str): Plot title.
        save_path (str): Path to save the plot.
    """
    data_a = np.array([[res[m] for m in metric_names] for res in results_a])
    data_b = np.array([[res[m] for m in metric_names] for res in results_b])

    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_names):
        plt.subplot(2, 2, i + 1)
        plt.boxplot([data_a[:, i], data_b[:, i]], labels=[label_a, label_b])
        plt.title(metric)
        plt.grid(True)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
