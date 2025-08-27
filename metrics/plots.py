#metrics/plots.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_boxplot(df, metric, group_col='model', ax=None, show=True, title=None):
    """
    Plots a boxplot of a metric grouped by a categorical column.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x=group_col, y=metric, data=df, ax=ax)
    ax.set_title(title or f'Distribution of {metric} by {group_col}')
    ax.set_ylabel(metric)
    ax.set_xlabel(group_col)

    if show:
        plt.tight_layout()
        plt.show()


def plot_effect_sizes(effect_dict, ax=None, show=True, title="Effect Sizes (Pillai's Trace)"):
    """
    Plots a bar chart of effect sizes per metric.
    """
    if not effect_dict:
        return

    metrics = list(effect_dict.keys())
    values = [effect_dict[m] for m in metrics]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(x=metrics, y=values, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Effect Size (Pillai's Trace)")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=45)

    if show:
        plt.tight_layout()
        plt.show()


def plot_posthoc_heatmap(pval_matrix, ax=None, show=True, title="Posthoc p-values (Tukey HSD)"):
    """
    Plots a heatmap of pairwise p-values for a given metric.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pval_matrix, annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if show:
        plt.tight_layout()
        plt.show()


def plot_normality_pvalues(results_dict, ax=None, show=True, title="Normality Test (Shapiro-Wilk p-values)"):
    """
    Plots Shapiro-Wilk test p-values per group.
    """
    groups = list(results_dict.keys())
    p_values = [results_dict[g]['p_value'] if results_dict[g]['p_value'] is not None else np.nan for g in groups]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(x=groups, y=p_values, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Shapiro-Wilk p-value")
    ax.set_xlabel("Group")
    plt.xticks(rotation=45)
    ax.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()


def plot_levene_result(result_dict):
    """
    Prints result of Levene's test and provides interpretation.
    """
    stat = result_dict.get("statistic")
    pval = result_dict.get("p_value")
    note = result_dict.get("note", "")

    if stat is None or pval is None:
        print(f"Levene's Test could not be computed: {note}")
    else:
        print(f"Levene’s Test statistic = {stat:.4f}, p-value = {pval:.4f}")
        if pval < 0.05:
            print("→ ⚠️ Variances are significantly different (heteroscedasticity).")
        else:
            print("→ ✅ Variances are not significantly different (homoscedasticity).")
