#metrics/run_per_metric_manova.py
import argparse
import pandas as pd
from pathlib import Path

from metrics.assumption_checks import check_multivariate_normality, check_box_m_test, check_vif
from metrics.statistical_tests import run_manova, run_posthoc_tests
from metrics.report_utils import save_dataframe
from metrics.plots import plot_group_distributions


def run_analysis(df, metrics, model_col='model', variant_col='variant', output_dir='manova_results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(f"\n📊 Running analysis for metric: {metric}")

        df_metric = df[[model_col, variant_col, metric]].dropna()

        # Step 1: Assumption Checks
        print("🔍 Checking assumptions...")
        check_multivariate_normality(df_metric, group_col=model_col, value_col=metric)
        check_box_m_test(df_metric, group_col=model_col, value_col=metric)
        check_vif(df_metric, feature_cols=[variant_col])

        # Step 2: MANOVA
        print("📈 Running MANOVA...")
        manova_df, eta_sq = run_manova(df_metric, metric=metric, group_col=model_col)
        save_dataframe(manova_df, f"{output_dir}/manova_{metric}", fmt="csv")

        # Step 3: Post-hoc Tests
        print("🧪 Running post-hoc tests...")
        posthoc_df = run_posthoc_tests(df_metric, metric=metric, group_col=model_col)
        save_dataframe(posthoc_df, f"{output_dir}/posthoc_{metric}", fmt="csv")

        # Step 4: Save Distribution Plots
        print("🖼️ Plotting distributions...")
        fig = plot_group_distributions(df_metric, group_col=model_col, value_col=metric, title=f"{metric} Comparison")
        fig_path = f"{output_dir}/plot_{metric}.png"
        fig.savefig(fig_path, dpi=300)
        print(f"✅ Plot saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MANOVA + Posthoc tests for each metric")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV containing experimental results")
    parser.add_argument("--metrics", nargs="+", required=True, help="List of metric column names to analyze")
    parser.add_argument("--model_col", type=str, default="model", help="Column name for model identifier")
    parser.add_argument("--variant_col", type=str, default="variant", help="Column name for variant identifier")
    parser.add_argument("--output_dir", type=str, default="manova_results", help="Directory to save outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    run_analysis(
        df,
        metrics=args.metrics,
        model_col=args.model_col,
        variant_col=args.variant_col,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
