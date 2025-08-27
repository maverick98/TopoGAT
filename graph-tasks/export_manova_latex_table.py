import os
import pandas as pd

def export_latex(dataset):
    logs_dir = f"/content/drive/MyDrive/{dataset}/manova_per_metric_logs"
    csv_path = os.path.join(logs_dir, f"{dataset}_manova_per_metric_summary.csv")
    df = pd.read_csv(csv_path)

    pivot_df = df.pivot(index="variant", columns="metric", values="p_value").round(4)
    latex_table = pivot_df.to_latex(index=True, caption=f"MANOVA p-values for {dataset}", label=f"tab:manova_{dataset}", escape=False)

    latex_path = os.path.join(logs_dir, f"{dataset}_manova_latex_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print(f"📄 LaTeX table written to: {latex_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name like MUTAG")
    args = parser.parse_args()
    export_latex(args.dataset)
