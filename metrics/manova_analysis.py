#metrics/manova_analysis.py
import os
import pandas as pd
import numpy as np
import logging
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']


def parse_summary_file(path):
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    mean_values = df['mean']
    return mean_values.to_dict()


def load_variant_data(variant_dir, model_family):
    data = {}
    for file in os.listdir(variant_dir):
        if file.startswith(f"summary_{model_family}"):
            model = model_family.upper()  # TopoGAT or TopoGIN
        elif file.startswith("summary_gat") and model_family == "topogat":
            model = "GAT"
        elif file.startswith("summary_gin") and model_family == "topogin":
            model = "GIN"
        else:
            continue

        full_path = os.path.join(variant_dir, file)
        try:
            stats = parse_summary_file(full_path)
            data[model] = stats
        except Exception as e:
            logger.warning(f"Failed to parse {full_path}: {e}")

    return data


def prepare_manova_dataset(dataset_root, model_family):
    rows = []

    for variant in os.listdir(dataset_root):
        variant_path = os.path.join(dataset_root, variant)
        if not os.path.isdir(variant_path):
            continue

        metrics_dict = load_variant_data(variant_path, model_family)

        for model_name, metrics in metrics_dict.items():
            row = {'variant': variant, 'model': model_name}
            row.update({metric: metrics.get(metric, np.nan) for metric in METRICS})
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.dropna()


def run_manova(df, dataset_name, model_family, output_dir):
    model_labels = df['model'].unique()
    assert len(model_labels) == 2, f"Expected 2 models, found: {model_labels}"

    formula = ' + '.join(METRICS) + ' ~ model'
    maov = MANOVA.from_formula(formula, data=df)
    result = maov.mv_test()

    out_path = os.path.join(output_dir, f'{dataset_name}_manova_results.txt')
    with open(out_path, 'w') as f:
        f.write(str(result))
    logger.info(f"MANOVA result saved to {out_path}")


def plot_metric_distributions(df, dataset_name, output_dir):
    melted = df.melt(id_vars=["model", "variant"], value_vars=METRICS,
                     var_name="Metric", value_name="Value")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x="Metric", y="Value", hue="model", palette="Set2")
    plt.title(f"Model Comparison on {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{dataset_name}_manova_boxplot.png")
    plt.savefig(plot_path)
    logger.info(f"Boxplot saved to {plot_path}")
    plt.close()


def analyze(dataset_name, model_family):
    assert model_family in ["topogat", "topogin"], "model_family must be 'topogat' or 'topogin'"

    dataset_root = f"/content/drive/MyDrive/{model_family}/{dataset_name}"
    if not os.path.exists(dataset_root):
        logger.error(f"Path does not exist: {dataset_root}")
        return

    logger.info(f"Analyzing {model_family.upper()} vs {'GAT' if model_family == 'topogat' else 'GIN'} on {dataset_name}")
    df = prepare_manova_dataset(dataset_root, model_family)
    if df.empty:
        logger.warning("No valid data found.")
        return

    run_manova(df, dataset_name, model_family, dataset_root)
    plot_metric_distributions(df, dataset_name, dataset_root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MANOVA between TopoGAT/TopoGIN and baseline GAT/GIN.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., MUTAG)")
    parser.add_argument("--model_family", required=True, choices=["topogat", "topogin"], help="Model family: topogat or topogin")
    args = parser.parse_args()

    analyze(args.dataset, args.model_family)
