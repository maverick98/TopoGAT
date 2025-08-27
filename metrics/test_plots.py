#metrics/test_plots.py
import pandas as pd
import numpy as np
from metrics import plots

# Simulate data for 3 models and 3 metrics
np.random.seed(42)
models = ['A', 'B', 'C']
metrics = ['accuracy', 'f1', 'precision']

rows = []
for model in models:
    for _ in range(30):
        row = {
            'model': model,
            'accuracy': np.random.normal(loc=0.8 + 0.05 * models.index(model), scale=0.02),
            'f1': np.random.normal(loc=0.75 + 0.04 * models.index(model), scale=0.03),
            'precision': np.random.normal(loc=0.70 + 0.03 * models.index(model), scale=0.025),
        }
        rows.append(row)

df = pd.DataFrame(rows)

# Plot boxplot for accuracy
plots.plot_boxplot(df, metric='accuracy', group_col='model', title='Test Boxplot: Accuracy by Model')

# Simulated effect sizes (e.g., Pillai's trace)
effect_sizes = {
    'accuracy': 0.21,
    'f1': 0.18,
    'precision': 0.14
}
plots.plot_effect_sizes(effect_sizes, title="Test Effect Sizes")

# Simulated posthoc p-value matrix
pval_matrix = pd.DataFrame({
    'A': [1.0, 0.03, 0.01],
    'B': [0.03, 1.0, 0.06],
    'C': [0.01, 0.06, 1.0]
}, index=['A', 'B', 'C'])

plots.plot_posthoc_heatmap(pval_matrix, title="Test Posthoc P-values (Tukey HSD)")
