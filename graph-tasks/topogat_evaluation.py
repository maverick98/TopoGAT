import sys
import os
import argparse
from utils.config import load_config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.topogat_vs_gat import TopoGATvsGATExperiment
from experiments.topogin_vs_gin import TopoGINvsGINExperiment


def perform_experiment(config, variant, base):
    if base.lower() == "gat":   
        run_experiment(TopoGATvsGATExperiment, config, variant, base)
    elif base.lower() == "gin":
        run_experiment(TopoGINvsGINExperiment, config, variant, base)
    else:
        raise ValueError(f"Unsupported base model '{base}'. Use 'gat' or 'gin'.")

def run_experiment(experiment_class, config, variant, base):
    print(f"Running {experiment_class.__name__} on dataset '{config['dataset']}' with variant '{variant}'")
    experiment = experiment_class(config=config, variant=variant)
    experiment.run()
    experiment.analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TopoGAT or TopoGIN experiments.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (e.g., MUTAG, PROTEINS, ENZYMES)")
    parser.add_argument("--variant", type=str, default="basic", help="Variant of Topo model to use (e.g., concat, attn)")
    parser.add_argument("--base", type=str, default="gat", help="Baseline model: 'gat' or 'gin'")
    args = parser.parse_args()

    config = load_config()
    config['dataset'] = args.dataset_name  # override dataset name from YAML

    perform_experiment(config, args.variant, args.base)
