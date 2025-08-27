import os
import re
import argparse
import pandas as pd

def extract_pvalue(text):
    match = re.search(r"Pr > F = ([\d\.eE-]+)", text)
    return float(match.group(1)) if match else None

def parse_manova_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    variant = os.path.basename(filepath).replace("_per_metric_manova.txt", "")
    current_metric = None
    data = []

    for line in lines:
        line = line.strip()
        if line.startswith("--- Metric:"):
            current_metric = line.split(":")[1].strip()
        elif "Pr > F" in line:
            p_value = extract_pvalue(line)
            data.append({
                "variant": variant,
                "metric": current_metric,
                "p_value": p_value,
                "significant_0.05": p_value < 0.05 if p_value is not None else None
            })

    return data

def main():
    parser = argparse.ArgumentParser(description="Summarize per-metric MANOVA results across variants")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., MUTAG)")
    args = parser.parse_args()

    logs_path = f"/content/drive/MyDrive/{args.dataset}/manova_per_metric_logs"
    all_files = [os.path.join(logs_path, f) for f in os.listdir(logs_path) if f.endswith(".txt")]

    all_results = []
    for file in all_files:
        all_results.extend(parse_manova_file(file))

    df = pd.DataFrame(all_results)
    output_csv = os.path.join(logs_path, f"{args.dataset}_manova_per_metric_summary.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved MANOVA summary to {output_csv}")

    try:
        # Also generate LaTeX table
        latex_table = df.pivot(index="variant", columns="metric", values="p_value")
        latex_output = os.path.join(logs_path, f"{args.dataset}_manova_summary.tex")
        with open(latex_output, "w") as f:
            f.write(latex_table.to_latex(float_format="%.4f", na_rep="--"))
        print(f"Saved LaTeX table to {latex_output}")
    except Exception as e:
        print(f"Could not export LaTeX table: {e}")

if __name__ == "__main__":
    main()
