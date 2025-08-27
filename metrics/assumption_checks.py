#metrics/assumption_checks.py
from pathlib import Path
import pandas as pd
from metrics.assumption import (
    check_normality_per_group,
    mardia_multivariate_normality,
    box_m_test,
    check_multicollinearity,
)


def check_multivariate_normality(df: pd.DataFrame,dependent_vars, group_col='model', output_path=None):
    """
    Runs both Shapiro-Wilk per group and Mardia's test.

    Returns:
        dict: {'shapiro': ..., 'mardia': ...}
    """
    #dependent_vars = [col for col in df.columns if col != group_col]
    data_matrix = df[dependent_vars].dropna().values

    shapiro_results = {}
    for metric in dependent_vars:
        shapiro_results[metric] = check_normality_per_group(df, metric_col=metric, group_col=group_col)

    mardia_results = {}
    try:
        mardia_results = mardia_multivariate_normality(data_matrix)
    except Exception as e:
        mardia_results = {'error': str(e)}

    results = {
        'shapiro': shapiro_results,
        'mardia': mardia_results,
    }

    if output_path:
        output_path = Path(output_path)
        (output_path / "normality_results.json").write_text(str(results))

    return results


def check_box_m_test(df: pd.DataFrame,dependent_vars, group_col='model', output_path=None):
    """
    Runs Box's M test by grouping the data and extracting feature matrices.
    """
    #dependent_vars = [col for col in df.columns if col != group_col]
    groups = {
        name: g[dependent_vars].dropna().values
        for name, g in df.groupby(group_col)
        if g[dependent_vars].dropna().shape[0] >= 2
    }
    print(f"groups = {groups}")
    result = box_m_test(groups)
    print(f"box_m_test result = {result}")

    if output_path:
        output_path = Path(output_path)
        (output_path / "box_m_results.json").write_text(str(result))

    return result


def check_multicollinearity_vif(df: pd.DataFrame, features,output_path=None, threshold=5.0):
    """
    Calculates VIF values and flags high multicollinearity.
    """
    #features = [col for col in df.columns if df[col].dtype.kind in 'fi']
    vif_data, high_vif = check_multicollinearity(df, features, threshold=threshold)

    result = {
        'vif': vif_data,
        'high_vif': high_vif
    }

    if output_path:
        output_path = Path(output_path)
        (output_path / "vif_results.json").write_text(str(result))

    return result
