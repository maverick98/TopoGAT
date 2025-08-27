import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def highlight_significance(val, alpha):
    if val < alpha:
        return 'background-color: lightgreen'
    else:
        return 'background-color: salmon'

def plot_colored_heatmap(csv_path, output_path, alpha):
    df = pd.read_csv(csv_path)
    pivot_df = df.pivot(index="variant", columns="metric", values="p_value")

    # Save styled Excel with conditional formatting
    styled = pivot_df.style.applymap(lambda v: highlight_significance(v, alpha)).format("{:.4f}")
    excel_path = output_path.replace(".png", f"_styled_alpha_{alpha}.xlsx")
    styled.to_excel(excel_path)
    print(f"✅ Excel with conditional formatting saved to: {excel_path}")

    # Static seaborn heatmap
    plt.figure(figsize=(10, max(3, len(pivot_df) * 0.6)))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'p-value'}, linewidths=0.5)
    plt.title(f"MANOVA p-values (α={alpha})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Static heatmap saved to: {output_path}")

def plot_interactive_heatmap(csv_path, html_output):
    df = pd.read_csv(csv_path)
    fig = px.imshow(
        df.pivot(index="variant", columns="metric", values="p_value"),
        text_auto=".4f",
        color_continuous_scale="YlGnBu",
        aspect="auto",
        labels=dict(color="p-value"),
        title="Interactive MANOVA Heatmap"
    )
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.write_html(html_output)
    print(f"✅ Interactive Plotly heatmap saved to: {html_output}")

def export_significant_hits(csv_path, alpha, output_path):
    df = pd.read_csv(csv_path)
    sig_df = df[df["p_value"] < alpha]
    sig_csv = output_path.replace(".png", f"_significant_alpha_{alpha}.csv")
    sig_df.to_csv(sig_csv, index=False)
    print(f"✅ Significant hits exported to: {sig_csv}")

def plot_per_metric_bars(csv_path, output_dir, alpha):
    df = pd.read_csv(csv_path)
    metrics = df["metric"].unique()

    for metric in metrics:
        metric_df = df[df["metric"] == metric].sort_values(by="p_value")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=metric_df, x="variant", y="p_value", palette="viridis")
        plt.axhline(alpha, color="red", linestyle="--", label=f"α = {alpha}")
        plt.title(f"p-values for {metric} (MANOVA)")
        plt.xticks(rotation=45)
        plt.ylabel("p-value")
        plt.legend()
        plt.tight_layout()
        file = os.path.join(output_dir, f"{metric}_pvalues_bar.png")
        plt.savefig(file)
        plt.close()
        print(f"📊 Bar chart for {metric} saved to: {file}")

def main():
    parser = argparse.ArgumentParser(description="Generate full MANOVA report from CSV")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., MUTAG")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default: 0.05)")
    args = parser.parse_args()

    logs_dir = f"/content/drive/MyDrive/{args.dataset}/manova_per_metric_logs"
    csv_file = os.path.join(logs_dir, f"{args.dataset}_manova_per_metric_summary.csv")

    heatmap_output = os.path.join(logs_dir, f"{args.dataset}_manova_heatmap.png")
    html_output = os.path.join(logs_dir, f"{args.dataset}_manova_heatmap.html")

    plot_colored_heatmap(csv_file, heatmap_output, args.alpha)
    plot_interactive_heatmap(csv_file, html_output)
    export_significant_hits(csv_file, args.alpha, heatmap_output)
    plot_per_metric_bars(csv_file, logs_dir, args.alpha)

if __name__ == "__main__":
    main()
