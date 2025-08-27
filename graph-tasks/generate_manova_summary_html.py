import os
import pandas as pd

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MANOVA Summary Report - {dataset}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}
        h1 {{ color: #2c3e50; }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

<h1>MANOVA Statistical Analysis Summary: {dataset}</h1>

<h2>📋 Overview Table of p-values</h2>
{summary_table}

<h2>🌡️ Heatmap of p-values</h2>
<img src="{heatmap_png}" alt="MANOVA Heatmap">

<h2>📊 Metric-wise p-value Bar Plots</h2>
{metric_bar_images}

<h2>🔗 Downloadable Artifacts</h2>
<ul>
    <li><a href="{interactive_html}" target="_blank">🔗 Interactive Heatmap (HTML)</a></li>
    <li><a href="{styled_excel}" target="_blank">📄 Excel with Highlights</a></li>
    <li><a href="{sig_csv}" target="_blank">✅ Significant p-values (CSV)</a></li>
</ul>

</body>
</html>
"""

def generate_html(dataset, logs_dir):
    csv_path = os.path.join(logs_dir, f"{dataset}_manova_per_metric_summary.csv")
    df = pd.read_csv(csv_path)
    summary_table = df.pivot(index="variant", columns="metric", values="p_value").round(4).to_html(classes="styled", border=1)

    # Collect image tags for bar plots
    metric_bar_images = ""
    for metric in df["metric"].unique():
        bar_img_path = f"{metric}_pvalues_bar.png"
        full_path = os.path.join(logs_dir, bar_img_path)
        if os.path.exists(full_path):
            metric_bar_images += f'<h3>{metric}</h3><img src="{bar_img_path}" alt="{metric} p-values">\n'

    html_filled = HTML_TEMPLATE.format(
        dataset=dataset,
        summary_table=summary_table,
        heatmap_png=f"{dataset}_manova_heatmap.png",
        interactive_html=f"{dataset}_manova_heatmap.html",
        styled_excel=f"{dataset}_manova_heatmap_styled_alpha_0.05.xlsx",
        sig_csv=f"{dataset}_manova_heatmap_significant_alpha_0.05.csv",
        metric_bar_images=metric_bar_images
    )

    output_file = os.path.join(logs_dir, f"{dataset}_manova_summary.html")
    with open(output_file, "w") as f:
        f.write(html_filled)

    print(f"✅ HTML summary report written to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name like MUTAG")
    args = parser.parse_args()
    logs_dir = f"/content/drive/MyDrive/{args.dataset}/manova_per_metric_logs"
    generate_html(args.dataset, logs_dir)
