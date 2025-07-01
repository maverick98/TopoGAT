# topogat_evaluation.py
# Evaluates TopoGAT vs GAT with statistical testing, visualizations, and CSV export
# Evaluates TopoGAT vs GIN with statistical testing, visualizations, and CSV export

import sys
import os
import argparse
import torch
import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.topogat_experiment import TopoGATvsGATExperiment
from experiments.topogat_experiment import TopoGATvsGINExperiment


def peform_topogat_vs_gat_experiment(dataset_name, variant):
    print(f"Running  TopoGATvsGATExperiment experiment on {dataset_name} with variant '{variant}'")
    experiment = TopoGATvsGATExperiment(dataset_name=dataset_name, variant=variant)
    experiment.run()
    experiment.analyze()
    
def peform_topogat_vs_gin_experiment(dataset_name, variant):
    print(f"Running  TopoGATvsGINExperiment experiment on {dataset_name} with variant '{variant}'")
    experiment = TopoGATvsGINExperiment(dataset_name=dataset_name, variant=variant)
    experiment.run()
    experiment.analyze()

def peform_experiment(dataset_name, variant,base):
    if base == "gat":
        peform_topogat_vs_gat_experiment(dataset_name, variant)
    if base == "gin":
        peform_topogat_vs_gin_experiment(dataset_name, variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TopoGAT experiments.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (e.g., MUTAG, PTC_MR,PROTEINS,ENZYMES)")
    parser.add_argument("--variant", type=str, default="basic", help="TopoGAT variant to use")
    parser.add_argument("--base", type=str, default="gat", help="Baseline variant to use")
    args = parser.parse_args()

    peform_experiment(args.dataset_name, args.variant , args.base)
