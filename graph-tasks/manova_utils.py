# manova_utils.py

import os
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from metrics.statistics import paired_cohens_d

# Common metric keys used everywhere
METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']

BASELINE_MAP = {
    'topogat': 'gat',
    'topogin': 'gin'
}

def load_csv(path, transpose=False, drop_summary=False):
    df = pd.read_csv(path, index_col=0 if transpose else None)
    if transpose:
        df = df.T
    if drop_summary and "summary" in df.columns:
        df = df.drop(columns=["summary"])
    return df

def add_model_label(df, label):
    df['model'] = label
    return df

def run_manova_from_df(df, metrics=None, group_col='model'):
    metrics = metrics or METRIC_KEYS
    formula = ' + '.join(metrics) + f' ~ {group_col}'
    return MANOVA.from_formula(formula, data=df).mv_test()

def run_scaled_manova(df1, df2, metrics=None):
    metrics = metrics or METRIC_KEYS
    scaler = StandardScaler()
    combined = pd.concat([df1, df2], axis=0)
    scaled = pd.DataFrame(scaler.fit_transform(combined[metrics]), columns=metrics)
    scaled['group'] = ['Topo'] * len(df1) + ['Base'] * len(df2)
    return run_manova_from_df(scaled, metrics, group_col='group')

def run_paired_tests(df1, df2, metrics=None):
    metrics = metrics or METRIC_KEYS
    results = {}
    for metric in metrics:
        vals1 = df1[metric].values
        vals2 = df2[metric].values
        t_stat, p_val = ttest_rel(vals1, vals2)
        d, ci_low, ci_high = paired_cohens_d(vals1, vals2)
        results[metric] = {
            't_stat': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'ci_low': ci_low,
            'ci_high': ci_high
        }
    return results

def collect_variants(dataset_dir):
    return [v for v in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, v))]
