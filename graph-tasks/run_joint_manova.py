import os
import argparse
import pandas as pd
from manova_utils import load_csv, add_model_label, run_manova_from_df, METRIC_KEYS, BASELINE_MAP

def collect_joint_dataset(dataset_dir, model_family, base_model):
    data = []
    for variant in os.listdir(dataset_dir):
        variant_path = os.path.join(dataset_dir, variant)
        if not os.path.isdir(variant_path): continue

        topo_file = os.path.join(variant_path, f"summary_{model_family}.csv")
        base_file = os.path.join(variant_path, f"summary_{base_model}.csv")

        if not os.path.exists(topo_file) or not os.path.exists(base_file):
            print(f"Skipping {variant}")
            continue

        topo = load_csv(topo_file, transpose=True, drop_summary=True)
        base = load_csv(base_file, transpose=True, drop_summary=True)

        for _, row in topo.iterrows():
            data.append({"model": model_family, "variant": variant, **row})
        for _, row in base.iterrows():
            data.append({"model": base_model, "variant": variant, **row})

    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--base_model", type=str)
    args = parser.parse_args()

    base_model = args.base_model or BASELINE_MAP[args.model_family]
    dataset_dir = os.path.join("/content/drive/MyDrive", args.dataset)
    df = collect_joint_dataset(dataset_dir, args.model_family, base_model)

    output_dir = os.path.join(dataset_dir, "manova_joint")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{args.model_family}_vs_{base_model}_combined.csv")
    txt_path = os.path.join(output_dir, f"{args.model_family}_vs_{base_model}_manova.txt")

    df.to_csv(csv_path, index=False)
    with open(txt_path, 'w') as f:
        f.write("===== MANOVA Results (Topo vs Base) =====\n")
        f.write(str(run_manova_from_df(df, METRIC_KEYS, group_col='model')))

    print(f"[Saved] {csv_path}, {txt_path}")

if __name__ == "__main__":
    main()
