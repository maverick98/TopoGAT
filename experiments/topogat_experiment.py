# experiments/topogat_experiment.py
import os
import torch
import numpy as np
import random
import wandb
from trainers.graph_trainer import GraphTrainer
from models.topogat_selector import get_topogat_model
from models.gat import BaseGAT
from models.gin import BaseGIN
from metrics.evaluator import (
    ModelEvaluator,
    summarize_experiment_results,
    pick_representative_seed,
    save_visuals_for_representative_seed,
    export_latex_table,
    export_csv_summary
   
)
from metrics.statistics import(
    paired_ttest_with_effect_size,
    plot_comparison_boxplots
)


class TopoGATvsGATExperiment:
    def __init__(self, dataset_name="MUTAG", variant="basic", seeds=20, use_wandb=False):
        self.dataset_name = dataset_name
        self.variant = variant
        self.seeds = list(range(seeds))
        self.results_topo = []
        self.results_gat = []
        self.models_topo = []
        self.models_gat = []
        self.loaders = []
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project="TopoGAT", name=f"TopoGAT_vs_GAT_{dataset_name}_{variant}")

    def get_results_base_dir(self):
        preferred_root = "/content/drive/MyDrive/topogat_vs_gat"
        if os.path.exists(preferred_root):
            base_dir = os.path.join(preferred_root, self.dataset_name, self.variant)
        else:
            base_dir = os.path.join("results", self.dataset_name, self.variant)
        os.makedirs(base_dir, exist_ok=True)
        print(f"[INFO] Results will be saved in: {os.path.abspath(base_dir)}")
        return base_dir

    def run(self):
        for seed in self.seeds:
            print(f"[INFO] Starting evaluation for seed {seed}")
            self._run_experiment_for_seed(seed)
            print(f"[INFO] Finished evaluation for seed {seed}")

    def _run_experiment_for_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        trainer = GraphTrainer(self.dataset_name, seed=seed)
        result_dir = os.path.join(self.get_results_base_dir(), f"seed_{seed}")
        os.makedirs(result_dir, exist_ok=True)

        # Run TopoGAT and GAT experiments for the seed
        print(f"[INFO] Evaluating for TOPOGAT variant {self.variant}")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=True)
        print(f"[INFO] Done evaluating for TOPOGAT variant {self.variant}")

        print(f"[INFO] Evaluating for GAT")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=False)
        print(f"[INFO] Done evaluating for GAT")

    def _evaluate_model_for_seed(self, trainer, result_dir, seed, use_topogat):
        if use_topogat:
            model = get_topogat_model(
                input_dim=trainer.sample.num_node_features,
                topo_dim=trainer.topo_dim,
                num_classes=trainer.dataset.num_classes,
                variant=self.variant
            )
            label = f"TopoGAT-{self.variant} (Seed {seed})"
            prefix = f"TopoGAT/seed_{seed}/"
            conf_path = os.path.join(result_dir, "confusion_topogat.png")
            pr_path = os.path.join(result_dir, "pr_topogat.png")
        else:
            model = BaseGAT(
                input_dim=trainer.sample.num_node_features,
                num_classes=trainer.dataset.num_classes
            )
            label = f"GAT (Seed {seed})"
            prefix = f"GAT/seed_{seed}/"
            conf_path = os.path.join(result_dir, "confusion_gat.png")
            pr_path = os.path.join(result_dir, "pr_gat.png")

        trainer.train_and_eval_model(model, label=label)
        evaluator = ModelEvaluator(model, trainer.test_loader)
        metrics = evaluator.evaluate(
            verbose=True,
            save_confusion_path=conf_path,
            save_pr_path=pr_path,
            log_to_wandb=self.use_wandb,
            wandb_prefix=prefix
        )

        if use_topogat:
            self.results_topo.append(metrics)
            self.models_topo.append(model)
        else:
            self.results_gat.append(metrics)
            self.models_gat.append(model)
        self.loaders.append(trainer.test_loader)

    def analyze(self):
        base_dir = self.get_results_base_dir()

        # Summarize and save
        summary_topo = summarize_experiment_results(self.results_topo,
                                                    output_csv_path=os.path.join(base_dir, "summary_topogat.csv"))
        summary_gat = summarize_experiment_results(self.results_gat,
                                                   output_csv_path=os.path.join(base_dir, "summary_gat.csv"))

        # Export LaTeX tables
        export_latex_table(summary_topo, os.path.join(base_dir, "topogat_table.tex"), model_name="TopoGAT")
        export_latex_table(summary_gat, os.path.join(base_dir, "gat_table.tex"), model_name="GAT")

        # Export CSV summary combined
        export_csv_summary(summary_topo, summary_gat, os.path.join(base_dir, "comparison_summary.csv"))

        # Select representative seed
        rep_idx_topo, rep_metrics_topo = pick_representative_seed(self.results_topo, metric='accuracy')
        rep_idx_gat, rep_metrics_gat = pick_representative_seed(self.results_gat, metric='accuracy')

        # Save plots for representative seeds
        save_visuals_for_representative_seed(self.models_topo[rep_idx_topo],
                                             self.loaders[rep_idx_topo],
                                             class_names=None,
                                             seed_label=f"topogat_rep_{rep_idx_topo}",
                                             output_dir=os.path.join(base_dir, f"seed_{self.seeds[rep_idx_topo]}"),
                                             log_to_wandb=self.use_wandb,
                                             wandb_prefix=f"TopoGAT/seed_{self.seeds[rep_idx_topo]}/")

        save_visuals_for_representative_seed(self.models_gat[rep_idx_gat],
                                             self.loaders[rep_idx_gat],
                                             class_names=None,
                                             seed_label=f"gat_rep_{rep_idx_gat}",
                                             output_dir=os.path.join(base_dir, f"seed_{self.seeds[rep_idx_gat]}"),
                                             log_to_wandb=self.use_wandb,
                                             wandb_prefix=f"GAT/seed_{self.seeds[rep_idx_gat]}/")

        # Paired t-tests and effect size
        stat_file = os.path.join(base_dir, "paired_ttest_results.txt")
        stats = paired_ttest_with_effect_size(self.results_topo, self.results_gat,
                                              metric_names=['accuracy', 'precision', 'recall', 'f1'])
        with open(stat_file, 'w') as f:
            for metric, values in stats.items():
                f.write(f"===== Paired t-tests on TopoGAT vs GAT for {metric} =====\n")
                f.write(f"T={values['t']:.4f}, p-value={values['p']:.4f}, Cohen's d={values['d']:.4f}\n\n")
        print(f"Saved statistical comparison to {stat_file}")

        # Plot boxplots
        plot_file = os.path.join(base_dir, "metric_comparison_boxplot.png")
        plot_comparison_boxplots(
            self.results_topo, self.results_gat,
            label_a=f"TopoGAT-{self.variant}",
            label_b="GAT",
            metric_names=['accuracy', 'precision', 'recall', 'f1'],
            title="TopoGAT vs GAT Comparison",
            save_path=plot_file
        )
        print(f"Saved boxplot to {plot_file}")

        if self.use_wandb:
            wandb.finish()

        print("\nAll analysis completed and saved.")






class TopoGATvsGINExperiment:
    def __init__(self, dataset_name="MUTAG", variant="basic", seeds=20, use_wandb=False):
        self.dataset_name = dataset_name
        self.variant = variant
        self.seeds = list(range(seeds))
        self.results_topo = []
        self.results_gin = []
        self.models_topo = []
        self.models_gin = []
        self.loaders = []
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project="TopoGAT", name=f"TopoGAT_vs_GIN_{dataset_name}_{variant}")

    def get_results_base_dir(self):
        preferred_root = "/content/drive/MyDrive/topogat_vs_gin"
        if os.path.exists(preferred_root):
            base_dir = os.path.join(preferred_root, self.dataset_name, f"{self.variant}_vs_gin")
        else:
            base_dir = os.path.join("results", self.dataset_name, f"{self.variant}_vs_gin")
        os.makedirs(base_dir, exist_ok=True)
        print(f"[INFO] Results will be saved in: {os.path.abspath(base_dir)}")
        return base_dir

    def run(self):
        for seed in self.seeds:
            print(f"[INFO] Starting evaluation for seed {seed}")
            self._run_experiment_for_seed(seed)
            print(f"[INFO] Finished evaluation for seed {seed}")

    def _run_experiment_for_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        trainer = GraphTrainer(self.dataset_name, seed=seed)
        result_dir = os.path.join(self.get_results_base_dir(), f"seed_{seed}")
        os.makedirs(result_dir, exist_ok=True)

        # Run TopoGAT and GIN experiments for the seed
        print(f"[INFO] Evaluating for TOPOGAT variant {self.variant}")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=True)
        print(f"[INFO] Done evaluating for TOPOGAT variant {self.variant}")
        
        print(f"[INFO] Evaluating for GIN")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=False)
        print(f"[INFO] Done evaluating for GIN")

    def _evaluate_model_for_seed(self, trainer, result_dir, seed, use_topogat):
        if use_topogat:
            model = get_topogat_model(
                input_dim=trainer.sample.num_node_features,
                topo_dim=trainer.topo_dim,
                num_classes=trainer.dataset.num_classes,
                variant=self.variant
            )
            label = f"TopoGAT-{self.variant} (Seed {seed})"
            prefix = f"TopoGAT/seed_{seed}/"
            conf_path = os.path.join(result_dir, "confusion_topogat.png")
            pr_path = os.path.join(result_dir, "pr_topogat.png")
        else:
            model = BaseGIN(
                input_dim=trainer.sample.num_node_features,
                num_classes=trainer.dataset.num_classes
            )
            label = f"GIN (Seed {seed})"
            prefix = f"GIN/seed_{seed}/"
            conf_path = os.path.join(result_dir, "confusion_gin.png")
            pr_path = os.path.join(result_dir, "pr_gin.png")

        trainer.train_and_eval_model(model, label=label)
        evaluator = ModelEvaluator(model, trainer.test_loader)
        metrics = evaluator.evaluate(
            verbose=True,
            save_confusion_path=conf_path,
            save_pr_path=pr_path,
            log_to_wandb=self.use_wandb,
            wandb_prefix=prefix
        )

        if use_topogat:
            self.results_topo.append(metrics)
            self.models_topo.append(model)
        else:
            self.results_gin.append(metrics)
            self.models_gin.append(model)

        self.loaders.append(trainer.test_loader)

    def analyze(self):
        base_dir = self.get_results_base_dir()

        # Summarize and save
        summary_topo = summarize_experiment_results(self.results_topo,
                                                    output_csv_path=os.path.join(base_dir, "summary_topogat.csv"))
        summary_gin = summarize_experiment_results(self.results_gin,
                                                   output_csv_path=os.path.join(base_dir, "summary_gin.csv"))

        # Export LaTeX tables
        export_latex_table(summary_topo, os.path.join(base_dir, "topogat_table.tex"), model_name="TopoGAT")
        export_latex_table(summary_gin, os.path.join(base_dir, "gin_table.tex"), model_name="GIN")

        # Export CSV summary combined
        export_csv_summary(summary_topo, summary_gin, os.path.join(base_dir, "comparison_summary.csv"))

        # Select representative seed
        rep_idx_topo, _ = pick_representative_seed(self.results_topo, metric='accuracy')
        rep_idx_gin, _ = pick_representative_seed(self.results_gin, metric='accuracy')

        # Save plots for representative seeds
        save_visuals_for_representative_seed(self.models_topo[rep_idx_topo],
                                             self.loaders[rep_idx_topo],
                                             class_names=None,
                                             seed_label=f"topogat_rep_{rep_idx_topo}",
                                             output_dir=os.path.join(base_dir, f"seed_{self.seeds[rep_idx_topo]}"),
                                             log_to_wandb=self.use_wandb,
                                             wandb_prefix=f"TopoGAT/seed_{self.seeds[rep_idx_topo]}/")

        save_visuals_for_representative_seed(self.models_gin[rep_idx_gin],
                                             self.loaders[rep_idx_gin],
                                             class_names=None,
                                             seed_label=f"gin_rep_{rep_idx_gin}",
                                             output_dir=os.path.join(base_dir, f"seed_{self.seeds[rep_idx_gin]}"),
                                             log_to_wandb=self.use_wandb,
                                             wandb_prefix=f"GIN/seed_{self.seeds[rep_idx_gin]}/")

        # Paired t-tests and effect size
        stat_file = os.path.join(base_dir, "paired_ttest_results.txt")
        stats = paired_ttest_with_effect_size(self.results_topo, self.results_gin,
                                              metric_names=['accuracy', 'precision', 'recall', 'f1'])
        with open(stat_file, 'w') as f:
            for metric, values in stats.items():
                f.write(f"===== Paired t-tests on TopoGAT vs GIN for {metric} =====\n")
                f.write(f"T={values['t']:.4f}, p-value={values['p']:.4f}, Cohen's d={values['d']:.4f}\n\n")
        print(f"Saved statistical comparison to {stat_file}")

        # Plot boxplots
        plot_file = os.path.join(base_dir, "metric_comparison_boxplot.png")
        plot_comparison_boxplots(
            self.results_topo, self.results_gin,
            label_a=f"TopoGAT-{self.variant}",
            label_b="GIN",
            metric_names=['accuracy', 'precision', 'recall', 'f1'],
            title="TopoGAT vs GIN Comparison",
            save_path=plot_file
        )
        print(f"Saved boxplot to {plot_file}")

        if self.use_wandb:
            wandb.finish()

        print("\nAll analysis completed and saved.")
