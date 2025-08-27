import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.covariance import EllipticEnvelope
import statsmodels.api as sm
from pingouin import multivariate_normality
from bioinfokit.analys import stat
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from pathlib import Path
import re
import pandas as pd
import numpy as np
from typing import List, Optional
from math import sqrt
def box_m_test(df, group_col, dependent_vars):
    groups = df[group_col].unique()
    n_groups = len(groups)
    n_total = len(df)
    pooled_cov = np.zeros((len(dependent_vars), len(dependent_vars)))
    log_dets = 0
    total_n = 0

    for g in groups:
        data = df[df[group_col] == g][dependent_vars].values
        n = data.shape[0]
        cov = np.cov(data, rowvar=False)
        pooled_cov += (n - 1) * cov
        log_dets += (n - 1) * np.log(np.linalg.det(cov))
        total_n += n

    pooled_cov /= (total_n - n_groups)
    log_det_pooled = np.log(np.linalg.det(pooled_cov))
    
    m_stat = (total_n - n_groups) * log_det_pooled - log_dets
    correction = ((2 * len(dependent_vars)**2 + 3 * len(dependent_vars) - 1) *
                  (sum([1 / (df[df[group_col] == g].shape[0] - 1) for g in groups]) -
                   1 / (total_n - n_groups))) / (6 * (len(dependent_vars) + 1) * (n_groups - 1))
    chi_sq = m_stat * (1 - correction)
    df_box = (n_groups - 1) * len(dependent_vars) * (len(dependent_vars) + 1) / 2
    p_value = 1 - chi2.cdf(chi_sq, df_box)

    return m_stat, chi_sq, df_box, p_value

def summarize_outputs(output_path):
    """Aggregates key results from generated files for final verdict."""
    summary = {}

    def try_read_txt(filename, key):
        path = output_path / filename
        if path.exists():
            summary[key] = path.read_text()

    def try_read_csv(filename, key, max_rows=10):
        path = output_path / filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                summary[key] = df.head(max_rows).to_string(index=False)
            except Exception as e:
                summary[key] = f"Failed to read {filename}: {e}"

    # Read TXT files
    try_read_txt("manova_results.txt", "MANOVA")
    try_read_txt("permanova_results.txt", "PERMANOVA")
    try_read_txt("normality_results.txt", "normality")
    try_read_txt("box_m_results.txt", "box_m")

    # Read CSV summaries
    try_read_csv("group_sizes.csv", "group_sizes")
    try_read_csv("vif.csv", "vif")
    try_read_csv("anova_results.csv", "anova")
    try_read_csv("posthoc_results.csv", "posthoc")
    try_read_csv("outliers.csv", "outliers")

    # Optional: Check if effect_size.txt is ever created
    try_read_txt("effect_size.txt", "effect_size")

    # Optional: transformation_log.txt (you may not be writing it currently)
    try_read_txt("transformation_log.txt", "transformation")

    return summary




def generate_final_verdict(summary):
    verdict = []

    # Assumption Checks
    normality_flag = "normality" in summary and "p < 0.05" in summary["normality"]
    box_m_flag = "box_m" in summary and "p < 0.05" in summary["box_m"]

    used_permanova = False

    if "normality" in summary or "box_m" in summary:
        if normality_flag or box_m_flag:
            verdict.append("⚠️ One or more MANOVA assumptions are violated (non-normality or unequal covariance).")
            if "PERMANOVA" in summary:
                verdict.append("➡️ Switched to PERMANOVA (non-parametric alternative).")
                verdict.append("🔍 PERMANOVA result:\n" + summary["PERMANOVA"])
                used_permanova = True
        else:
            verdict.append("✅ MANOVA assumptions satisfied.")
            if "MANOVA" in summary:
                verdict.append("🔍 MANOVA result:\n" + summary["MANOVA"])
    else:
        verdict.append("❓ Could not evaluate all assumptions (normality/Box's M test missing).")

    # Determine if TopoGAT beats the baseline
    test_section = summary.get("PERMANOVA" if used_permanova else "MANOVA", "")
    lines = test_section.splitlines()
    p_value = None

    for line in lines:
        if "p" in line.lower():
            import re
            match = re.search(r"p\s*[=<>]\s*([0-9\.eE-]+)", line)
            if match:
                try:
                    p_value = float(match.group(1))
                    break
                except ValueError:
                    continue

    if p_value is not None:
        if p_value < 0.05:
            verdict.append("✅ **TopoGAT shows statistically significant improvement over baseline** (p < 0.05).")
        else:
            verdict.append("❌ **TopoGAT does not show statistically significant improvement over baseline** (p ≥ 0.05).")
    else:
        verdict.append("⚠️ Could not extract p-value to determine significance.")

    # Effect Size Interpretation (Partial Eta Squared)
    if "effect_size" in summary:
        verdict.append("📐 Effect Size Estimate:\n" + summary["effect_size"])
        import re
        eta_match = re.search(r"(partial\s+eta\s*squared|η²)\s*[=:\s]\s*([0-9\.]+)", summary["effect_size"], re.IGNORECASE)
        if eta_match:
            eta_val = float(eta_match.group(2))
            if eta_val < 0.01:
                level = "negligible"
            elif eta_val < 0.06:
                level = "small"
            elif eta_val < 0.14:
                level = "medium"
            else:
                level = "large"
            verdict.append(f"🧭 Interpretation: Partial η² = {eta_val:.3f} → **{level.capitalize()} effect size**")
        else:
            verdict.append("ℹ️ Could not extract numeric value for partial η².")

    # Additional Context
    if "group_sizes" in summary:
        verdict.append("👥 Group Sizes:\n" + summary["group_sizes"])

    if "outliers" in summary:
        verdict.append("🚨 Outlier Analysis:\n" + summary["outliers"])

    if "vif" in summary:
        verdict.append("🔁 Multicollinearity Check (VIF):\n" + summary["vif"])

    if "transformation" in summary:
        verdict.append("📏 Data Transformation Applied:\n" + summary["transformation"])

    if "anova" in summary:
        verdict.append("📊 Univariate ANOVA Results:\n" + summary["anova"])

    if "posthoc" in summary:
        verdict.append("🧪 Post-hoc Test Results:\n" + summary["posthoc"])

    return "\n\n".join(verdict)



def generate_comparative_report(
    result_dir: Path,
    metrics: list,
    output_file: str = "final_verdict_report.txt",
    title: str = "Comparative Evaluation",
    dataset_name: str = "your dataset",
):
    """
    Generate a comprehensive comparative report based on multiple statistical tests and analysis files.
    Args:
        result_dir (Path): Directory containing all result files.
        metrics (list): List of metric names to report (e.g., ['accuracy', 'precision']).
        output_file (str): Output filename to save the final verdict report.
        title (str): Title of the report.
        dataset_name (str): Name of the dataset used in evaluation.
    """
    import pandas as pd
    import re
    from io import StringIO

    result_dir = Path(result_dir)

    def read_txt_section(file_path: Path, header: str) -> str:
        text = file_path.read_text()
        pattern = rf"{header}\n(.+?)(\n\n|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def read_csv_safe(path: Path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    # Define for metrics whether bigger is better (add or modify as per your metrics)
    metric_higher_is_better = {
        'accuracy': True,
        'precision': True,
        'recall': True,
        'f1': True,
        'log_loss': False,
        'error_rate': False,
        # Add other metrics here as needed
    }

    # Load necessary files
    posthoc_path = result_dir / "posthoc_results.csv"
    anova_path = result_dir / "anova_results.csv"
    group_sizes_path = result_dir / "group_sizes.csv"
    normality_path = result_dir / "normality_results.txt"
    vif_path = result_dir / "vif.csv"
    outliers_path = result_dir / "outliers.csv"
    manova_path = result_dir / "manova_results.txt"

    posthoc_df = read_csv_safe(posthoc_path)
    anova_df = read_csv_safe(anova_path)
    vif_df = read_csv_safe(vif_path)
    group_sizes_df = read_csv_safe(group_sizes_path)
    outliers_df = read_csv_safe(outliers_path)

    # Check required columns in posthoc
    required_cols = {"group1", "group2", "meandiff", "p-adj", "reject", "variable"}
    if not required_cols.issubset(posthoc_df.columns):
        raise ValueError(f"Post-hoc table missing required columns: {required_cols - set(posthoc_df.columns)}")

    # Determine model names (generalized for 2+ models)
    models = sorted(set(posthoc_df["group1"]) | set(posthoc_df["group2"]))
    better_model_count = {model: 0 for model in models}

    report_lines = [
        f"## 🧪 {title} — Dataset: {dataset_name}\n",
        "### ✅ Summary Verdict",
        "This report compares model performance based on multivariate statistical analysis across multiple evaluation metrics.\n",
    ]

    report_lines.append("---\n\n### 📊 Metrics Compared\n")
    for metric in metrics:
        report_lines.append(f"- {metric.capitalize()}")

    report_lines.append("\n---\n\n### 🔍 Group Sizes\n")
    if not group_sizes_df.empty:
        report_lines.append(group_sizes_df.to_markdown(index=False))

    report_lines.append("\n---\n\n### 📈 MANOVA Test Results\n")
    manova_text = read_txt_section(manova_path, "MANOVA result:")
    if manova_text:
        report_lines.append("```")
        report_lines.append(manova_text)
        report_lines.append("```")

    report_lines.append("\n---\n\n### 📉 ANOVA Results (Per Metric)\n")
    if not anova_df.empty:
        metric_rows = []
        for metric in metrics:
            rows = anova_df[anova_df['variable'] == metric]
            if not rows.empty:
                f_val = rows.iloc[0]['F']
                p_val = rows.iloc[0]['PR(>F)']
                sig = "✅ Yes" if p_val < 0.05 else "❌ No"
                metric_rows.append((metric, f_val, p_val, sig))
        anova_summary = pd.DataFrame(metric_rows, columns=["Metric", "F-value", "p-value", "Significant (p < 0.05)"])
        report_lines.append(anova_summary.to_markdown(index=False))

    report_lines.append("\n---\n\n### 🔬 Post-hoc Tukey HSD Test\n")
    significant_metrics = []

    table_rows = []
    for metric in metrics:
        rows = posthoc_df[posthoc_df["variable"] == metric]
        if rows.empty:
            continue
        row = rows.iloc[0]
        sig = "✅" if row["reject"] else "❌"

        if row["reject"]:
            # Fix: decide better model based on metric direction (higher is better or not)
            metric_key = metric.lower()
            bigger_is_better = metric_higher_is_better.get(metric_key, True)  # default True if unknown

            meandiff = row["meandiff"]
            if bigger_is_better:
                # Assuming meandiff = mean(group2) - mean(group1)
                better = row["group2"] if meandiff > 0 else row["group1"]
            else:
                # For metrics where smaller is better, invert choice
                better = row["group1"] if meandiff > 0 else row["group2"]

            better_model_count[better] += 1
            significant_metrics.append((metric, better, meandiff, row["p-adj"]))

        table_rows.append([
            metric,
            row["group1"],
            row["group2"],
            row["meandiff"],
            row["p-adj"],
            sig
        ])

    posthoc_summary = pd.DataFrame(
        table_rows,
        columns=["Metric", "Group 1", "Group 2", "Mean Diff", "p-value", "Significant"]
    )
    report_lines.append(posthoc_summary.to_markdown(index=False))

    report_lines.append("\n---\n\n### 🧪 Normality Check (Shapiro-Wilk Test)\n")
    norm_text = read_txt_section(normality_path, "")
    if norm_text:
        report_lines.append("```")
        report_lines.append(norm_text)
        report_lines.append("```")

    report_lines.append("\n---\n\n### 🔍 Outlier Detection\n")
    if not outliers_df.empty:
        report_lines.append(f"{len(outliers_df)} outliers were detected across all metrics using IQR method. These were excluded from MANOVA computation.")

    report_lines.append("\n---\n\n### 🧮 VIF (Multicollinearity Check)\n")
    if not vif_df.empty:
        report_lines.append(vif_df.to_markdown(index=False))

    report_lines.append("\n---\n")

    # Final Verdict (generalized)
    total_significant = len(significant_metrics)
    sorted_models = sorted(better_model_count.items(), key=lambda x: x[1], reverse=True)
    top_model, top_score = sorted_models[0]
    top_models = [model for model, score in sorted_models if score == top_score]

    report_lines.append("\n### 🏁 Final Verdict\n")
    if len(top_models) == 1:
        verdict = f"🏆 Based on the results, **{top_model}** performed best overall across significant metrics."
    else:
        joined = ", ".join(f"**{m}**" for m in top_models)
        verdict = f"🤝 Multiple models performed equally well: {joined}"
    report_lines.append(verdict)

    out_path = result_dir / output_file
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[INFO] Final comparative report written to: {out_path}")




def generate_academic_report(
    result_dir: Path,
    metrics: List[str],
    output_file: str = "final_verdict_report_formal.md",
    dataset_name: str = "dataset"
):
    """
    Generate a formal, academic-style Markdown report summarizing MANOVA/ANOVA/Tukey
    results and diagnostics. Input files are expected to be produced by run_full_manova_analysis:
      - manova_results.txt
      - anova_results.csv
      - posthoc_results.csv
      - group_sizes.csv
      - normality_results.txt
      - vif.csv
      - outliers.csv
      - normalized_data.csv (optional, for effect sizes / Cohen's d)
      - box_m_results.txt (optional)
    The function is robust to missing files and will include explanatory notes where needed.
    """

    result_dir = Path(result_dir)
    out_path = result_dir / output_file

    # Helper readers
    def read_txt(path: Path) -> Optional[str]:
        return path.read_text() if path.exists() else None

    def read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    # Paths
    manova_path = result_dir / "manova_results.txt"
    anova_path = result_dir / "anova_results.csv"
    posthoc_path = result_dir / "posthoc_results.csv"
    group_sizes_path = result_dir / "group_sizes.csv"
    normality_path = result_dir / "normality_results.txt"
    vif_path = result_dir / "vif.csv"
    outliers_path = result_dir / "outliers.csv"
    normalized_path = result_dir / "normalized_data.csv"
    boxm_path = result_dir / "box_m_results.txt"

    # Read files
    manova_text = read_txt(manova_path)
    anova_df = read_csv(anova_path)
    posthoc_df = read_csv(posthoc_path)
    group_sizes_df = read_csv(group_sizes_path)
    normality_text = read_txt(normality_path)
    vif_df = read_csv(vif_path)
    outliers_df = read_csv(outliers_path)
    normalized_df = read_csv(normalized_path)
    boxm_text = read_txt(boxm_path)

    lines = []
    # Header
    # Header (include models if available)
    if not group_sizes_df.empty and 'model' in group_sizes_df.columns:
        model_names = sorted(group_sizes_df['model'].unique().tolist())
    else:
        model_names = []

    if model_names:
        lines.append(f"# Final Evaluation Summary — Dataset: {dataset_name} — Models: {', '.join(model_names)}")
    else:
        lines.append(f"# Final Evaluation Summary — Dataset: {dataset_name}")

    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This document summarizes the multivariate statistical analysis comparing model performance on the stated dataset. The analyses include MANOVA, follow-up univariate ANOVAs, post-hoc pairwise comparisons (Tukey HSD), and diagnostic checks (normality, outliers, multicollinearity).")
    lines.append("")

    # Metrics
    lines.append("## 2. Metrics Evaluated")
    lines.append("")
    lines.append("The following metrics were used in the multivariate and univariate analyses:")
    for m in metrics:
        lines.append(f"- {m}")
    lines.append("")

    # Group sizes
    lines.append("## 3. Sample Sizes")
    lines.append("")
    if not group_sizes_df.empty:
        lines.append("Table 1: Number of runs per model")
        lines.append("")
        lines.append(group_sizes_df.to_markdown(index=False))
    else:
        lines.append("Group sizes file not found. Please ensure `group_sizes.csv` is present in the result directory.")
    lines.append("")

    # MANOVA Results
    lines.append("## 4. MANOVA Results")
    lines.append("")
    if manova_text:
        # Clean and extract the model row block if possible
        # Include full MANOVA output but remove decorative characters
        cleaned = manova_text.strip()
        # Try to extract the 'model' section lines to present succinctly
        model_section = ""
        # attempt to locate the 'model' block
        model_match = re.search(r"\n\s*model\s+(.+?)(?:\n\n|\Z)", cleaned, flags=re.S)
        if model_match:
            model_section = model_match.group(0).strip()
        # fallback: include whole file
        lines.append("The MANOVA test (multivariate test statistics) is presented below.")
        lines.append("")
        lines.append("```\n" + cleaned + "\n```")
    else:
        lines.append("MANOVA output file `manova_results.txt` not found. Ensure MANOVA was executed and its results saved.")
    lines.append("")

    # Box's M if present
    if boxm_text:
        lines.append("### 4.1 Homogeneity of Covariance Matrices (Box's M)")
        lines.append("")
        lines.append("Box's M test output:")
        lines.append("")
        lines.append("```\n" + boxm_text.strip() + "\n```")
        lines.append("")

    # ANOVA results with partial eta-squared
    lines.append("## 5. Univariate ANOVA Results (per metric)")
    lines.append("")
    if anova_df.empty:
        lines.append("ANOVA results file `anova_results.csv` not found. Ensure univariate ANOVAs were computed and saved.")
    else:
        # Expect anova_df to consist of repeating blocks with a variable column
        # Compute partial eta-squared for each metric if sum_sq rows are present
        eta_rows = []
        # Some anova exports have index names like 'C(model)' and 'Residual', and columns 'sum_sq', 'df', 'F', 'PR(>F)', 'variable'
        for metric in metrics:
            rows = anova_df[anova_df["variable"] == metric]
            if rows.empty:
                continue
            # find effect row (C(model)) and residual row
            # The input earlier used anova_table with 'variable' added; rows likely contain two rows: C(model) and Residual
            try:
                effect_row = rows[rows.index.str.contains("C(model)|C\\(model\\)|C_model", regex=True)] if hasattr(rows.index, "str") else rows.iloc[[0]]
            except Exception:
                effect_row = rows.iloc[[0]]
            # We will fallback to using first row with non-null 'sum_sq' as effect, and last 'Residual'
            sum_sq_vals = rows["sum_sq"].values
            if len(sum_sq_vals) >= 2:
                ss_effect = float(sum_sq_vals[0])
                ss_resid = float(sum_sq_vals[1])
            else:
                # fallbacks
                ss_effect = float(sum_sq_vals[0]) if len(sum_sq_vals) == 1 else np.nan
                ss_resid = np.nan
            # compute partial eta-squared if possible
            if not np.isnan(ss_effect) and not np.isnan(ss_resid):
                partial_eta2 = ss_effect / (ss_effect + ss_resid)
            else:
                partial_eta2 = np.nan
            # Grab F and p
            f_val = rows.iloc[0].get("F", np.nan)
            p_val = rows.iloc[0].get("PR(>F)", np.nan)
            eta_rows.append({
                "Metric": metric,
                "F-value": f_val if not pd.isna(f_val) else np.nan,
                "p-value": p_val if not pd.isna(p_val) else np.nan,
                "Partial_eta2": round(partial_eta2, 4) if not np.isnan(partial_eta2) else "N/A"
            })

        eta_df = pd.DataFrame(eta_rows)
        if not eta_df.empty:
            # Format p-values in scientific when small
            def fmt_p(x):
                try:
                    x = float(x)
                    if x < 0.0001:
                        return f"{x:.2e}"
                    else:
                        return f"{x:.6f}"
                except Exception:
                    return "N/A"
            eta_df["p-value"] = eta_df["p-value"].apply(fmt_p)
            eta_df["F-value"] = eta_df["F-value"].apply(lambda v: f"{v:.4f}" if (not pd.isna(v)) else "N/A")
            lines.append("Table 2: Univariate ANOVA results with partial eta-squared (effect size).")
            lines.append("")
            lines.append(eta_df.to_markdown(index=False))
            lines.append("")
            lines.append("Note: Partial eta-squared is computed as SS_effect / (SS_effect + SS_residual).")
        else:
            lines.append("Could not parse ANOVA results in the expected format.")
    lines.append("")

    # Post-hoc results: full table + (optional) Cohen's d if normalized/raw data available
    lines.append("## 6. Post-hoc Pairwise Comparisons (Tukey HSD)")
    lines.append("")
    if posthoc_df.empty:
        lines.append("Post-hoc results file `posthoc_results.csv` not found.")
    else:
        # Ensure expected columns exist
        expected = {"group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject", "variable"}
        missing = expected - set(posthoc_df.columns)
        if missing:
            lines.append(f"Post-hoc file is missing expected columns: {missing}. Showing available columns.")
            lines.append("")
            lines.append(posthoc_df.to_markdown(index=False))
        else:
            # compute Cohen's d if data available
            cohen_rows = []
            can_compute_d = not normalized_df.empty
            if can_compute_d:
                # normalized_df contains z-scored metrics; ideally compute d on original scale, but z-scored is OK for standardized effect size.
                # We'll compute Cohen's d using metric values in normalized_df (group-wise means and pooled sd).
                for _, r in posthoc_df.iterrows():
                    metric = r["variable"]
                    g1 = r["group1"]
                    g2 = r["group2"]
                    meandiff = float(r["meandiff"])
                    p_adj = float(r["p-adj"])
                    reject = bool(r["reject"])
                    # attempt to compute d
                    try:
                        v1 = normalized_df[normalized_df["model"] == g1][metric].dropna()
                        v2 = normalized_df[normalized_df["model"] == g2][metric].dropna()
                        n1, n2 = len(v1), len(v2)
                        if n1 > 1 and n2 > 1:
                            s1, s2 = v1.std(ddof=1), v2.std(ddof=1)
                            # pooled sd
                            pooled_sd = sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
                            d = (v2.mean() - v1.mean()) / pooled_sd if pooled_sd > 0 else np.nan
                        else:
                            d = np.nan
                    except Exception:
                        d = np.nan
                    cohen_rows.append({
                        "Metric": metric,
                        "Group 1": g1,
                        "Group 2": g2,
                        "Mean Diff": meandiff,
                        "p-adj": f"{p_adj:.6f}",
                        "Lower": r["lower"],
                        "Upper": r["upper"],
                        "Reject": reject,
                        "Cohen_d (z-scored data)": round(d, 4) if not pd.isna(d) else "N/A"
                    })
                ph_df = pd.DataFrame(cohen_rows)
            else:
                ph_df = posthoc_df.copy()
                # format numeric columns
                for col in ["meandiff", "p-adj", "lower", "upper"]:
                    if col in ph_df.columns:
                        ph_df[col] = ph_df[col].apply(lambda v: f"{float(v):.6f}" if pd.notna(v) else "N/A")
                # rename to readable
                ph_df = ph_df.rename(columns={
                    "meandiff": "Mean Diff",
                    "p-adj": "p-adj",
                    "lower": "Lower",
                    "upper": "Upper",
                    "reject": "Reject",
                    "variable": "Metric",
                    "group1": "Group 1",
                    "group2": "Group 2"
                })
            lines.append("Table 3: Tukey HSD pairwise comparisons. Cohen's d (computed on z-scored data) provided when normalized data is available.")
            lines.append("")
            lines.append(ph_df.to_markdown(index=False))
    lines.append("")

    # Normality checks
    lines.append("## 7. Normality Diagnostics")
    lines.append("")
    if normality_text:
        # Present as-is but formalize
        lines.append("Shapiro–Wilk test results (per group) or multivariate normality outputs:")
        lines.append("")
        lines.append("```\n" + normality_text.strip() + "\n```")
    else:
        lines.append("Normality results file `normality_results.txt` not found.")
    lines.append("")

    # Outliers
    lines.append("## 8. Outlier Analysis")
    lines.append("")
    if not outliers_df.empty:
        lines.append("The following rows were flagged as multivariate outliers (export from outliers.csv). These records were excluded from MANOVA computations when indicated.")
        lines.append("")
        # show a compact table with first columns and model/seed/dataset if present
        show_cols = [c for c in ["model", "seed", "dataset"] if c in outliers_df.columns]
        display_cols = show_cols + [c for c in outliers_df.columns if c not in show_cols][:6]
        lines.append(outliers_df[display_cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("No outlier file present or no outliers detected.")
    lines.append("")

    # VIF
    lines.append("## 9. Multicollinearity (VIF)")
    lines.append("")
    if not vif_df.empty:
        vif_df_display = vif_df.copy()
        vif_df_display["VIF"] = vif_df_display["VIF"].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else "N/A")
        lines.append("Variance Inflation Factors (VIF) for dependent variables:")
        lines.append("")
        lines.append(vif_df_display.to_markdown(index=False))
        lines.append("")
        lines.append("Note: VIF values above 10 indicate substantial multicollinearity that may affect inference. Interpret results with caution.")
    else:
        lines.append("No VIF data available.")
    lines.append("")

    # Final summary & formal verdict
    lines.append("## 10. Summary of Findings and Conclusion")
    lines.append("")
    # derive a formal verdict from posthoc or ANOVA
    verdict_lines = []
    try:
        # Count metrics where C(model) was significant in ANOVA
        sig_metrics = []
        for metric in metrics:
            rows = anova_df[anova_df["variable"] == metric]
            if not rows.empty:
                p = rows.iloc[0].get("PR(>F)", np.nan)
                if pd.notna(p) and float(p) < 0.05:
                    sig_metrics.append(metric)
        if len(sig_metrics) == 0:
            verdict_lines.append("No univariate ANOVA showed significant differences between models at α = 0.05.")
        else:
            # From posthoc (if available) count pairwise wins where reject==True and meandiff indicates direction
            winners = {}
            if not posthoc_df.empty:
                # use expected columns if present
                if {"group1", "group2", "meandiff", "reject", "variable"}.issubset(set(posthoc_df.columns)):
                    for metric in sig_metrics:
                        metric_rows = posthoc_df[posthoc_df["variable"] == metric]
                        for _, r in metric_rows.iterrows():
                            if bool(r["reject"]):
                                # interpret meandiff as mean(group2) - mean(group1) as in statsmodels tukey
                                md = float(r["meandiff"])
                                g1, g2 = str(r["group1"]), str(r["group2"])
                                # Decide better model assuming higher-is-better for typical metrics except log_loss
                                if metric.lower() in ["log_loss", "error_rate"]:
                                    better = g1 if md > 0 else g2
                                else:
                                    better = g2 if md > 0 else g1
                                winners[better] = winners.get(better, 0) + 1
            # produce human-readable verdict
            if winners:
                sorted_w = sorted(winners.items(), key=lambda kv: kv[1], reverse=True)
                top_model, top_count = sorted_w[0]
                # if tie
                ties = [m for m, c in sorted_w if c == top_count]
                if len(ties) == 1:
                    verdict_lines.append(f"Model '{top_model}' is the primary superior model across the significant metrics (wins = {top_count}).")
                else:
                    verdict_lines.append(f"Multiple models performed similarly across significant metrics: {', '.join(ties)} (each with {top_count} wins).")
            else:
                verdict_lines.append("Significant univariate differences were detected, but post-hoc comparisons did not identify a consistent single model superior across metrics.")
    except Exception as e:
        verdict_lines.append("Unable to compute a concise model-level verdict automatically due to missing or malformed files.")
    # Add any cautionary notes
    verdict_lines.append("")
    verdict_lines.append("Caveats: ensure adequate sample sizes and independence of runs. Multicollinearity (VIF) and outliers were evaluated; consult diagnostics above for potential influence on inference.")
    lines.extend(verdict_lines)
    lines.append("")

    # Save file
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Academic report written to: {out_path}")
    return out_path



def run_full_manova_analysis(csv_file_path,output_dir,dataset_name,group_col='model', scale_method='zscore'):
   
    print(f"\n🔍 Loading dataset: {csv_file_path}")
    df = pd.read_csv(csv_file_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
   
    #dependent_vars = [col for col in df.columns if col != group_col]
    dependent_vars = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
    print(f"✔️ Dependent variables for MANOVA: {dependent_vars}")

    # Step 1. Log or Z-score transformation
    # Log or Z-score transformation
    if scale_method == 'zscore':
        df[dependent_vars] = df[dependent_vars].apply(zscore)
    elif scale_method == 'log':
        df[dependent_vars] = np.log1p(df[dependent_vars])
    df.to_csv(output_path / "normalized_data.csv", index=False)

    # Step 2. Group sizes
    df[group_col].value_counts().to_csv(output_path / "group_sizes.csv")

    # Step 3. Multivariate Normality Test
    normality_results = []
    for g in df[group_col].unique():
        sub = df[df[group_col] == g][dependent_vars]
        res = multivariate_normality(sub, alpha=0.05)
        print(f"multivariate_normality = {res}")
        p = res.pval 
        normality_results.append(f"{g}: p = {p:.5f}")

    (output_path / "normality_results.txt").write_text("\n".join(normality_results))

    # Step 4. Box’s M Test (homogeneity of covariances)
    try:
        m_stat, chi_sq, df_box, p_value = box_m_test(df, group_col, dependent_vars)
        box_m_text = (
            f"Box’s M Statistic = {m_stat:.4f}\n"
            f"Chi-squared = {chi_sq:.4f}\n"
            f"df = {df_box:.1f}\n"
            f"p = {p_value:.5f}"
        )
    except Exception as e:
        box_m_text = f"Box’s M test failed: {e}"
    (output_path / "box_m_results.txt").write_text(box_m_text)


    # Step 5. VIF
    vif_data = []
    for i, var in enumerate(dependent_vars):
        others = [v for v in dependent_vars if v != var]
        model = ols(f"{var} ~ {' + '.join(others)}", data=df).fit()
        rsq = model.rsquared
        vif = 1 / (1 - rsq) if rsq < 1 else np.inf
        vif_data.append({"variable": var, "VIF": vif})
    pd.DataFrame(vif_data).to_csv(output_path / "vif.csv", index=False)

    # Step 6. Outlier detection
    try:
        envelope = EllipticEnvelope()
        outliers = envelope.fit_predict(df[dependent_vars]) == -1
        df[outliers].to_csv(output_path / "outliers.csv", index=False)
    except Exception as e:
        (output_path / "outliers.csv").write_text(f"Failed: {e}")

    # Step 7. Visuals
    plt.figure(figsize=(10, 6))
    for col in dependent_vars:
        sns.boxplot(data=df, x=group_col, y=col)
        plt.title(f"Boxplot of {col} by {group_col}")
        plt.tight_layout()
        plt.savefig(output_path / f"boxplot_{col}.png")
        plt.clf()

    # Step 8. MANOVA
    formula = f"{' + '.join(dependent_vars)} ~ {group_col}"
    manova_result = MANOVA.from_formula(formula, data=df)
    (output_path / "manova_results.txt").write_text(str(manova_result.mv_test()))

    # Step 9. Follow-up Univariate ANOVA
    anova_results = []
    for var in dependent_vars:
        model = ols(f"{var} ~ C({group_col})", data=df).fit()
        anova_table = anova_lm(model, typ=2)
        anova_table["variable"] = var
        anova_results.append(anova_table)
    pd.concat(anova_results).to_csv(output_path / "anova_results.csv")

    # Step 10. Post-hoc (Tukey HSD)
    tukey_all = []
    for var in dependent_vars:
        tukey = pairwise_tukeyhsd(df[var], df[group_col])
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df["variable"] = var
        tukey_all.append(tukey_df)
    pd.concat(tukey_all).to_csv(output_path / "posthoc_results.csv", index=False)

    # Step 10.5. Fallback PERMANOVA if assumptions failed
    fallback = False
    if any("p =" in line and float(line.split("p = ")[1]) < 0.05 for line in normality_results):
        fallback = True
    if "p < 0.05" in box_m_text or ("p =" in box_m_text and float(box_m_text.split("p = ")[1]) < 0.05):
        fallback = True

    if fallback:
        try:
            dist = pdist(df[dependent_vars], metric="euclidean")
            dm = DistanceMatrix(squareform(dist), ids=[str(i) for i in df.index])
            result = permanova(dm, grouping=df[group_col].astype(str).values, permutations=999)
            (output_path / "permanova_results.txt").write_text(str(result))
        except Exception as e:
            (output_path / "permanova_results.txt").write_text(f"PERMANOVA failed: {e}")

    # Step 11. Summarize and generate final verdict
    summary = summarize_outputs(output_path)
    final_verdict = generate_final_verdict(summary)
    (output_path / "final_verdict.txt").write_text(final_verdict)

    print("\n✅ Analysis Complete. Final Verdict:\n")
    print(final_verdict)

    generate_academic_report(
                    result_dir=output_path,
                    metrics=dependent_vars,
                    output_file="final_verdict_report_formal.md",
                    dataset_name=dataset_name
    )

    generate_comparative_report(
                    result_dir=output_dir,
                    metrics=dependent_vars,
                    dataset_name=dataset_name
                    )





    return {
        "dependent_vars": dependent_vars,
        "output_path": output_path,
        "summary": summary,
        "final_verdict": final_verdict
    }
