#metrics/statistics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, shapiro, levene
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
from utils.logger import get_logger

logger = get_logger(__name__)

def run_statistical_tests(df_topo, df_base, metric_names):
    results = []
    for metric in metric_names:
        a = pd.DataFrame(df_topo)[metric]
        b = pd.DataFrame(df_base)[metric]
        try:
            t_stat, p_val = ttest_rel(a, b)
            d, d_lo, d_hi = paired_cohens_d(a, b)
            results.append({
                "metric": metric,
                "t": t_stat,
                "p_value": p_val,
                "effect_size": d,
                "d_ci_low": d_lo,
                "d_ci_high": d_hi
            })
        except Exception as e:
            logger.error(f"[run_statistical_tests] Failed for metric '{metric}': {e}")
            results.append({
                "metric": metric,
                "p_value": np.nan,
                "effect_size": np.nan
            })
    return results


# ---------- EFFECT SIZE ----------

def paired_cohens_d(x, y, confidence=0.95):
    """Cohen's d with confidence interval for paired samples"""
    x, y = np.array(x), np.array(y)
    diff = x - y
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    d = mean_diff / std_diff if std_diff > 0 else 0.0

    # CI
    se = std_diff / np.sqrt(n)
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    ci = t_crit * se
    return d, d - ci, d + ci

# ---------- T-TEST ----------

def paired_ttest_with_effect_size(results_a, results_b, metric_names=None):
    """Paired t-test with Cohen’s d for each metric"""
    if not results_a or not results_b:
        logger.warning("Empty results provided for t-test.")
        return {}

    if metric_names is None:
        metric_names = list(results_a[0].keys())

    stats = {}
    for metric in metric_names:
        try:
            a = np.array([res[metric] for res in results_a])
            b = np.array([res[metric] for res in results_b])
            t_stat, p_val = ttest_rel(a, b)
            d, d_lo, d_hi = paired_cohens_d(a, b)
            stats[metric] = {
                "t": t_stat, "p": p_val,
                "d": d, "d_ci_low": d_lo, "d_ci_high": d_hi
            }
        except Exception as e:
            logger.error(f"Failed t-test for metric '{metric}': {e}")
    return stats

# ---------- MANOVA ----------

def run_manova(df: pd.DataFrame, metric: str, variant: str, base_model: str) -> dict:
    """Run MANOVA on a given metric between a variant and base model."""
    df_metric = df[['model', metric]].dropna().copy()
    df_metric = df_metric.rename(columns={metric: 'value'})
    df_metric['metric'] = metric

    pivot_df = df_metric.pivot(columns='model', values='value')
    if variant not in pivot_df.columns or base_model not in pivot_df.columns:
        return {"metric": metric, "variant": variant, "p_value": np.nan, "effect_size": np.nan}

    valid = pivot_df[[variant, base_model]].dropna()
    if valid.empty or len(valid) < 4:
        return {"metric": metric, "variant": variant, "p_value": np.nan, "effect_size": np.nan}

    analysis_df = valid[[variant, base_model]]
    analysis_df.columns = ['x1', 'x2']
    formula = 'x1 + x2 ~ 1'

    try:
        manova = MANOVA.from_formula(formula, data=analysis_df)
        result = manova.mv_test()
        wilks_p = result.results['Intercept']['stat'].loc["Wilks' lambda", 'Pr > F']
        eta_sq = compute_partial_eta_squared(analysis_df)
        normality_p = max(shapiro(analysis_df['x1']).pvalue,
                          shapiro(analysis_df['x2']).pvalue)
        homogeneity_p = levene(analysis_df['x1'], analysis_df['x2']).pvalue
    except Exception as e:
        logger.error(f"[MANOVA Error] {metric} / {variant}: {e}")
        return {"metric": metric, "variant": variant, "p_value": np.nan, "effect_size": np.nan}

    return {
        "metric": metric,
        "variant": variant,
        "p_value": wilks_p,
        "effect_size": eta_sq,
        "normality_p": normality_p,
        "homogeneity_p": homogeneity_p,
        "n": len(valid)
    }

def compute_partial_eta_squared(df: pd.DataFrame) -> float:
    """Approximate partial eta squared from two dependent groups."""
    diff = df['x1'] - df['x2']
    ss_effect = np.sum((diff - np.mean(diff))**2)
    ss_total = np.sum((df['x1'] - np.mean(df['x1']))**2) + np.sum((df['x2'] - np.mean(df['x2']))**2)
    return ss_effect / (ss_effect + ss_total + 1e-9) if ss_total > 0 else 0.0

# ---------- T-TEST STANDALONE ----------

def run_ttest(df: pd.DataFrame, metric: str, variant: str, base_model: str) -> dict:
    """Run paired t-test on a given metric between a variant and base model."""
    df_metric = df[['model', metric]].dropna().copy()
    df_metric = df_metric.rename(columns={metric: 'value'})
    df_metric['metric'] = metric

    pivot_df = df_metric.pivot(columns='model', values='value')
    if variant not in pivot_df.columns or base_model not in pivot_df.columns:
        return {"metric": metric, "variant": variant, "p_value": np.nan}

    valid = pivot_df[[variant, base_model]].dropna()
    if valid.empty:
        return {"metric": metric, "variant": variant, "p_value": np.nan}

    try:
        t_stat, p_value = ttest_rel(valid[variant], valid[base_model])
    except Exception as e:
        logger.error(f"[T-Test Error] {metric} / {variant}: {e}")
        return {"metric": metric, "variant": variant, "p_value": np.nan}

    return {"metric": metric, "variant": variant, "p_value": p_value}

# ---------- FDR CORRECTION ----------

def apply_fdr_correction(results: list, alpha=0.05) -> list:
    """Applies FDR correction on p-values and adds `significant` field."""
    p_vals = [r['p_value'] for r in results]
    mask = ~np.isnan(p_vals)
    corrected = results.copy()

    if np.sum(mask) > 0:
        _, pvals_corrected, _, rejected = multipletests(
            p_vals, alpha=alpha, method='fdr_bh'
        )
        for i, r in enumerate(corrected):
            r['p_value_corrected'] = pvals_corrected[i] if not np.isnan(p_vals[i]) else np.nan
            r['significant'] = bool(rejected[i]) if not np.isnan(p_vals[i]) else False
    else:
        for r in corrected:
            r['p_value_corrected'] = np.nan
            r['significant'] = False

    return corrected

# ---------- VISUALIZATION ----------

import matplotlib.pyplot as plt

def plot_comparison_boxplots(results_topo, results_base, metrics, save_path=None):
    import pandas as pd

    df_topo = pd.DataFrame(results_topo)
    df_base = pd.DataFrame(results_base)
    df_topo['model'] = 'Topo'
    df_base['model'] = 'Base'

    df_all = pd.concat([df_topo, df_base], ignore_index=True)

    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        df_all.boxplot(column=metric, by='model', ax=axs[i])
        axs[i].set_title(metric)
        axs[i].set_xlabel('')
        axs[i].set_ylabel(metric)

    plt.suptitle("Metric Comparison")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

