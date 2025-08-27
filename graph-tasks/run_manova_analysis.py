import argparse
import subprocess
import os
from manova_utils import BASELINE_MAP, collect_variants, load_csv, run_scaled_manova, run_paired_tests, METRIC_KEYS

def call_subprocess(script, args_dict):
    cmd = ["python", script] + [f"--{k}={v}" for k, v in args_dict.items()]
    print(f"[Calling] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_internal_tests(dataset_name, model_family):
    base_model = BASELINE_MAP[model_family]
    dataset_dir = f"/content/drive/MyDrive/{dataset_name}"

    for variant in collect_variants(dataset_dir):
        variant_path = os.path.join(dataset_dir, variant)
        topo_csv = os.path.join(variant_path, f"raw_scores_{model_family}.csv")
        base_csv = os.path.join(variant_path, f"raw_scores_{base_model}.csv")

        if not os.path.exists(topo_csv) or not os.path.exists(base_csv):
            print(f"Skipping {variant} (missing files)")
            continue

        topo_df = load_csv(topo_csv)
        base_df = load_csv(base_csv)
        log_path = os.path.join(variant_path, f"manova_statistical_analysis_{model_family}_vs_{base_model}.log")

        with open(log_path, 'w') as f:
            f.write("===== Scaled MANOVA =====\n")
            f.write(str(run_scaled_manova(topo_df, base_df)) + "\n\n")

            f.write("===== Paired T-Test + Cohen's d =====\n")
            test_results = run_paired_tests(topo_df, base_df)
            for metric, res in test_results.items():
                f.write(f"{metric}: t={res['t_stat']:.4f}, p={res['p_value']:.4f}, "
                        f"d={res['cohens_d']:.4f}, CI=({res['ci_low']:.4f}, {res['ci_high']:.4f})\n")

        print(f"[Done] {variant} → {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_family", type=str, choices=["topogat", "topogin"], required=True)
    args = parser.parse_args()

    base_model = BASELINE_MAP[args.model_family]

    call_subprocess("run_per_variant_manova.py", {
        "dataset": args.dataset,
        "model_family": args.model_family
    })

    call_subprocess("run_joint_manova.py", {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "base_model": base_model
    })

    run_internal_tests(args.dataset, args.model_family)

if __name__ == "__main__":
    main()
