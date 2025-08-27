#metrics/statistical_tests.py
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_manova(df, dependent_vars, factor_col='model'):
    """
    Runs MANOVA using statsmodels and returns key test statistics:
    - Wilks’ Lambda
    - Pillai’s Trace
    - Hotelling-Lawley Trace
    - Roy’s Greatest Root
    - Approximate effect size via Pillai’s Trace
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        dependent_vars (list): List of metric columns to test
        factor_col (str): Categorical column name representing group

    Returns:
        dict: Test statistics with stat and p-values
    """
    try:
        formula = ' + '.join(dependent_vars) + ' ~ ' + factor_col
        manova = MANOVA.from_formula(formula, data=df)
        result = manova.mv_test()

        stats = {}
        for stat_name in ['Wilks\' lambda', 'Pillai\'s trace',
                          'Hotelling-Lawley trace', 'Roy\'s greatest root']:
            try:
                stat_val = result.results[factor_col]['stat'].loc[stat_name, 'Value']
                p_val = result.results[factor_col]['stat'].loc[stat_name, 'Pr > F']
                stats[stat_name] = {'stat': stat_val, 'p': p_val}
            except Exception:
                stats[stat_name] = {'stat': None, 'p': None}

        # Effect size approximation using Pillai's Trace
        stats['partial_eta_squared'] = stats.get("Pillai's trace", {}).get('stat', None)

        return stats

    except Exception as e:
        return {'error': str(e)}


def run_posthoc_tukey(df, dependent_var, factor_col='model', alpha=0.05):
    """
    Runs Tukey HSD posthoc test for a given dependent variable.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        dependent_var (str): The metric/column to test
        factor_col (str): Grouping column (categorical)
        alpha (float): Significance level

    Returns:
        pd.DataFrame: Summary of Tukey HSD results
    """
    try:
        tukey = pairwise_tukeyhsd(endog=df[dependent_var], groups=df[factor_col], alpha=alpha)
        df_res = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        return df_res
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})


def fdr_correction(results, alpha=0.05):
    """
    Applies Benjamini-Hochberg FDR correction to a list of result dicts.

    Parameters:
        results (list of dict): Each dict must have a 'p_value' key
        alpha (float): Desired FDR level (default 0.05)

    Returns:
        list of dict: Updated results with 'p_value_corrected' and 'significant'
    """
    p_vals = [r.get('p_value') for r in results]
    mask = [p is not None and not np.isnan(p) for p in p_vals]

    if not any(mask):
        for r in results:
            r['p_value_corrected'] = None
            r['significant'] = False
        return results

    pvals_to_correct = np.array([p_vals[i] for i, valid in enumerate(mask) if valid])
    reject, pvals_corrected, _, _ = multipletests(pvals_to_correct, alpha=alpha, method='fdr_bh')

    idx = 0
    for i, valid in enumerate(mask):
        if valid:
            results[i]['p_value_corrected'] = pvals_corrected[idx]
            results[i]['significant'] = bool(reject[idx])
            idx += 1
        else:
            results[i]['p_value_corrected'] = None
            results[i]['significant'] = False

    return results
