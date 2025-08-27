#metrics/__init__.py
from .core import compute_classification_metrics
from .evaluator import (
    ModelEvaluator,
    summarize_experiment_results,
    export_latex_table,
    export_csv_summary,
    pick_representative_seed,
    save_visuals_for_representative_seed
)
from .statistics import (
    run_statistical_tests,
    paired_cohens_d,
    paired_ttest_with_effect_size,
    run_manova,
    compute_partial_eta_squared,
    run_ttest,
    apply_fdr_correction,
    plot_comparison_boxplots
)

# You can include plot functions if you plan to use them
try:
    from .plots import plot_confusion, plot_precision_recall
except ImportError:
    pass  # optional import

__all__ = [
    "compute_classification_metrics",
    "ModelEvaluator",
    "summarize_experiment_results",
    "export_latex_table",
    "export_csv_summary",
    "pick_representative_seed",
    "save_visuals_for_representative_seed",
    "cohen_d",
    "get_confidence_interval",
    "compute_t_statistic",
    "format_stat_result",
    "aggregate_statistics",
    "report_statistics",
    "run_manova",
]
