import os
import argparse
import pandas as pd
from manova_utils import load_csv, add_model_label, METRIC_KEYS, BASELINE_MAP

from statsmodels.multivariate.manova import MANOVA

def run_manova_per_metric(df, metric):
    df_metric = df[['model', metric]].dropna()
    df_metric = df_metric.rename(columns={metric: 'value'})
    df_metric['metric'] = metric
    return MANOVA.from_formula('value ~ model', data=df_metric).mv_test()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_family", type=str, choices=["topogat", "topogin"], required=True)
    args = parser.parse_args()

    dataset_dir = f"/content/drive/MyDrive/{args.dataset}"
    base_model = BASELINE_MAP[args.model_family]
    output_dir = os.path.join(dataset_dir, "manova_per_metric_logs")
    os.makedirs(output_dir, exist_ok=True)

    all_variants = [v for v in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, v))]

    for variant in all_variants:
        variant_path = os.path.join(dataset_dir, variant)
        topo_file = os.path.join(variant_path, f"raw_{args.model_family}.csv")
        base_file = os.path.join(variant_path, f"raw_{base_model}.csv")

        if not os.path.exists(topo_file) or not os.path.exists(base_file):
            print(f"[Skipping] {variant}")
            continue

        topo_df = add_model_label(pd.read_csv(topo_file), args.model_family.upper())
        base_df = add_model_label(pd.read_csv(base_file), base_model.upper())
        combined = pd.concat([topo_df, base_df], ignore_index=True)

        log_path = os.path.join(output_dir, f"{variant}_per_metric_manova.txt")
        with open(log_path, "w") as f:
            f.write(f"=== MANOVA (Per-Metric) Results for {variant} ===\n\n")
            for metric in METRIC_KEYS:
                result = run_manova_per_metric(combined, metric)
                f.write(f"--- Metric: {metric} ---\n{str(result)}\n\n")
        print(f"[Done] {variant} → {log_path}")

if __name__ == "__main__":
    main()
