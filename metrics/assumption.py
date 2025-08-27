#metrics/assumption.py
import pandas as pd
import numpy as np
from scipy.stats import shapiro, chi2
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_normality_per_group(df: pd.DataFrame, metric_col: str, group_col: str = 'model'):
    """
    Applies Shapiro-Wilk normality test per group.
    """
    print(f"metric_col={metric_col}")
    print(f"group_col={group_col}")
    
    results = {}
    for group_name, group_df in df.groupby(group_col):
        data = group_df[metric_col].dropna()
        if len(data) < 3:  # Shapiro requires at least 3 samples
            results[group_name] = {'statistic': None, 'p_value': None, 'error': 'Too few samples'}
        else:
            stat, p = shapiro(data)
            results[group_name] = {'statistic': stat, 'p_value': p}
    return results

def mardia_multivariate_normality(X: np.ndarray):
    """
    Mardia's multivariate normality test.
    """
    n, p = X.shape
    X_centered = X - np.mean(X, axis=0)
    S = np.cov(X_centered, rowvar=False)
    S_inv = np.linalg.inv(S)

    b1_p = np.mean([((x @ S_inv @ x.T) ** 2) for x in X_centered])
    b2_p = np.sum([((x @ S_inv @ x.T)) for x in X_centered]) / n

    skewness = n * b1_p / 6
    kurtosis = (b2_p - p) / (np.sqrt(8 * p))

    skew_p = 1 - chi2.cdf(skewness, df=p*(p+1)*(p+2)/6)
    kurt_p = 2 * (1 - chi2.cdf(kurtosis**2, df=1))

    return {
        'skewness': skewness,
        'skew_p_value': skew_p,
        'kurtosis_z': kurtosis,
        'kurt_p_value': kurt_p,
    }

def box_m_test(groups: dict):
    """
    Performs Box's M test for equality of covariance matrices across groups.
    Each group must be a numpy array (n_i, p).
    """
    n_groups = len(groups)
    n_total = sum(g.shape[0] for g in groups.values())
    p = next(iter(groups.values())).shape[1]

    cov_matrices = {k: np.cov(v, rowvar=False) for k, v in groups.items()}
    pooled_cov = sum([(g.shape[0] - 1) * cov for g, cov in zip(groups.values(), cov_matrices.values())]) / (n_total - n_groups)

    M_stat = 0
    for k, X in groups.items():
        n_k = X.shape[0]
        cov_k = cov_matrices[k]
        term = (n_k - 1) * np.log(np.linalg.det(cov_k))
        M_stat += term
    pooled_term = (n_total - n_groups) * np.log(np.linalg.det(pooled_cov))
    M_stat = pooled_term - M_stat

    c = ((2 * p**2 + 3 * p - 1) * (n_groups - 1)) / (6 * (sum([1/(X.shape[0]-1) for X in groups.values()]) - 1))
    df = (p * (p + 1) * (n_groups - 1)) / 2

    p_value = 1 - chi2.cdf(M_stat * (1 - c), df)

    return {
        'M_stat': M_stat,
        'p_value': p_value,
        'degrees_of_freedom': df
    }

def check_multicollinearity(df: pd.DataFrame, features: list, threshold: float = 5.0):
    """
    Calculates VIF and flags high multicollinearity.
    """
    X = df[features].dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif = vif_data[vif_data["VIF"] > threshold]
    return vif_data.to_dict(orient='records'), high_vif.to_dict(orient='records')
