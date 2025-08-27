import pandas as pd
from metrics.report_utils import generate_latex_table

def test_generate_latex_table():
    df = pd.DataFrame({
        "Metric": ["Accuracy", "F1-score"],
        "GAT Mean": [0.83, 0.81],
        "TopoGAT Mean": [0.86, 0.84],
        "p-value": [0.045, 0.031]
    })
    
    latex = generate_latex_table(df, caption="TopoGAT vs GAT Results", label="tab:topogat_vs_gat")
    
    assert "\\begin{table}" in latex
    assert "TopoGAT Mean" in latex
    assert "\\caption{" in latex
    assert "\\label{" in latex
    assert "\\end{table}" in latex
