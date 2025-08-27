#metrics/report_utils.py
import pandas as pd
import os
from pathlib import Path


def save_dataframe(df, path, fmt='csv', index=False, float_format="%.4f"):
    """
    Save a DataFrame to CSV or LaTeX.

    Parameters:
        df (pd.DataFrame): The DataFrame to save
        path (str): File path to save to (extension optional)
        fmt (str): 'csv' or 'latex'
        index (bool): Whether to write row names
        float_format (str): Float format for LaTeX/CSV
    """
    path = Path(path)
    if fmt == 'csv':
        if not path.name.endswith('.csv'):
            path = path.with_suffix('.csv')
        df.to_csv(path, index=index, float_format=float_format)
        print(f"✅ Saved CSV: {path}")
    elif fmt == 'latex':
        if not path.name.endswith('.tex'):
            path = path.with_suffix('.tex')
        with open(path, 'w') as f:
            latex = df.to_latex(index=index, float_format=float_format)
            f.write(latex)
        print(f"✅ Saved LaTeX: {path}")
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def generate_summary_table(stats_dict):
    """
    Converts a nested dictionary of statistics into a summary table.

    Parameters:
        stats_dict (dict): Nested dictionary of form
            {metric: {model: value}}

    Returns:
        pd.DataFrame: Formatted summary table
    """
    return pd.DataFrame(stats_dict).T


def save_plot(fig, path, dpi=300):
    """
    Save a matplotlib figure.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save
        path (str): Path to save to (extension determines format)
        dpi (int): Dots per inch
    """
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved plot to {path}")
