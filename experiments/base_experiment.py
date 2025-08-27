#experiments/base_experiment.py
import os
import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
from trainers.graph_trainer import GraphTrainer
from metrics.manova_runner import run_full_manova_analysis
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict, Counter
from metrics.evaluator import (
    ModelEvaluator,
    summarize_experiment_results,
    pick_representative_seed,
    save_visuals_for_representative_seed,
    export_latex_table,
    export_csv_summary
   
)
class BaseTopoComparisonExperiment:
    def __init__(self,config,final_report,force_train_eval,topo_model,variant,base_model_name,seeds,base_model_fn,topo_model_fn,output_dir):
        print(f"[INFO ] config = {config}")
        self.config=config
        self.final_report=final_report
        self.force_train_eval=force_train_eval
        self.topo_model=topo_model
        self.variant=variant
        self.base_model_name=base_model_name
        self.dataset = config["dataset"]
        self.seeds = list(range(seeds))
        self.base_model_fn=base_model_fn
        self.topo_model_fn=topo_model_fn
        self.output_dir=output_dir
        self.results_topo = []
        self.results_gat = []
        self.results_baseline = []
        self.models_topo = []
        self.models_gat = []
        self.models_baseline = []
        self.loaders = []
    def get_all_datasets_dir(self):
        dataset_dir = "/content/drive/MyDrive/topology/results/"+self.topo_model
        return  dataset_dir      
    def get_preferred_root(self):
        preferred_root = "/content/drive/MyDrive/topology/results/"+self.output_dir
        return  preferred_root
    def does_preferred_root_exist(self):
        preferred_root_exists=False;
        preferred_root = self.get_preferred_root()
        if os.path.exists(preferred_root):
            preferred_root_exists=True
        return preferred_root_exists
    def get_results_base_dir(self, subdir=""):
        #if self.does_preferred_root_exist() == True:
        base_dir = self.get_preferred_root()
        #else:
        #    base_dir = os.path.join("results", self.dataset, subdir)
        os.makedirs(base_dir, exist_ok=True)
        print(f"[INFO] Results will be saved in: {os.path.abspath(base_dir)}")
        return base_dir
    def get_metrics_dir(self):
        return os.path.join(self.get_results_base_dir(), "all_metrics")        
    def get_dataset_result_csv(self):
        dataset_result_csv=None
        csv_output_dir = Path(self.get_metrics_dir())
        if os.path.exists(csv_output_dir):
            dataset_result_csv = csv_output_dir / f"{self.dataset}_{self.variant}.csv"
            
        return  dataset_result_csv        

    def is_dataset_result_present(self):
        dataset_result_present=False
        preferred_root_exists=self.does_preferred_root_exist()
        if  preferred_root_exists == False:
            return preferred_root_exists
        csv_path = self.get_dataset_result_csv()
        if csv_path is not None:
            dataset_result_present=True
        return  dataset_result_present
    def should_train(self):
        print(f'Checking if training and evaluation required or not?')
        training_reqd =True
        if self.force_train_eval ==1:
            training_reqd=True
        else:
            if self.does_preferred_root_exist():
                training_reqd=False
        return training_reqd        

    
    def save_metric_data(self):
        print(f'After running all seeds — Save results')
        all_results = self.results_topo + self.results_gat
        df = pd.DataFrame(all_results)
        csv_output_dir = Path(self.get_metrics_dir())
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.get_dataset_result_csv()
        print(f"\n📄 Combined metrics being saved  to: {csv_path}")
        df.to_csv(csv_path, index=False)
        #print(f"\n📄 Combined metrics saved to: {csv_path.resolve()}")
    def execute_all_seeds(self):
        should_run=self.should_train()
        if should_run == True:
            print(f"Starting Training and Evaluation...")
            for seed in self.seeds:
                print(f"[INFO] -----------------START-------------------")
                print(f"[INFO] Starting evaluation for seed {seed}")
                self._run_experiment_for_seed(seed)
                print(f"[INFO] Finished evaluation for seed {seed}")
                print(f"[INFO] -----------------END--------------------")

            self.save_metric_data()
        else:    
            print(f'Training and Evaluation not required')          
        return should_run    

    def run(self):
        #Executing all seeds
        self.execute_all_seeds()
        #Run MANOVA analysis now
        self.analyze()

    def _run_experiment_for_seed(self, seed):
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Delegate dataset/model loading to trainer
        trainer = GraphTrainer(self.config, seed=seed)
        result_dir = os.path.join(self.get_results_base_dir(), f"seed_{seed}")
        os.makedirs(result_dir, exist_ok=True)

        # Subclass must implement both model setups
        print(f"[INFO] Evaluating TopoModel for seed {seed}")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=True)
        print(f"[INFO] Finished evaluating TopoModel for seed {seed}")
        print(f"[INFO] .............................................")
        print(f"[INFO] Evaluating Baseline model for seed {seed}")
        self._evaluate_model_for_seed(trainer, result_dir, seed, use_topogat=False)

    def _evaluate_model_for_seed(self, trainer, result_dir, seed, use_topogat):
        input_dim=trainer.sample.num_node_features
        topo_dim=trainer.topo_dim
        num_classes=trainer.dataset.num_classes
        print(f'input_dim :: {input_dim}')
        print(f'topo_dim :: {topo_dim}')
        print(f'num_classes :: {num_classes}')
        if use_topogat:
            model = self.topo_model_fn(input_dim=input_dim,
                                    topo_dim=topo_dim,
                                    num_classes=num_classes)
            label = f"TopoGAT-{self.variant} (Seed {seed})"
            prefix = f"TopoGAT/seed_{seed}/"
            conf_path = os.path.join(result_dir, "confusion_topogat.png")
            pr_path = os.path.join(result_dir, "pr_topogat.png")
        else:
            model = self.base_model_fn(
                input_dim=trainer.sample.num_node_features,
                num_classes=num_classes
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
        metrics["model"] = self.variant if use_topogat else self.base_model_name
        metrics["seed"] = seed
        metrics["dataset"] = self.dataset

        print(f'metrics is ::: {metrics}')
        if use_topogat:
            self.results_topo.append(metrics)
            self.models_topo.append(model)
        else:
            self.results_gat.append(metrics)
            self.models_gat.append(model)
        self.loaders.append(trainer.test_loader)







    def generate_final_winner_report_with_metrics_auto(
        self,
        dataset_name: str,
        metrics: list,
        results_root: str,
        output_file: str = "final_dataset_winner_report.txt",
    ):
        """
        Auto-discovers all variant comparisons for a dataset and aggregates winner and metrics.
        Args:
            dataset_name (str): Name of the dataset (e.g., 'MUTAG')
            metrics (list): List of metrics to consider
            results_root (str): Root folder containing all dataset result folders
            output_file (str): Output summary report filename
        """
        base_dir = Path(results_root) / dataset_name
        if not base_dir.exists():
            print(f"[ERROR] Results folder for dataset not found: {base_dir}")
            return

        variant_result_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not variant_result_dirs:
            print(f"[ERROR] No variant result folders found in: {base_dir}")
            return

        win_counter = Counter()
        metric_sums = defaultdict(lambda: defaultdict(float))
        metric_counts = defaultdict(lambda: defaultdict(int))
        model_names = set()

        for result_dir in variant_result_dirs:
            report_path = result_dir / "all_metrics"  / "final_verdict_report.txt"
            metrics_path = result_dir / "all_metrics" / f"{dataset_name}_{result_dir.name}.csv"
            variant_name = result_dir.name
            winner_found = False

            # 1. Extract winner from final_verdict_report.txt if available
            if report_path.exists():
                text = report_path.read_text()
                match = re.search(r"Final Verdict.*?\*\*(.*?)\*\*", text, re.DOTALL)
                if match:
                    winner = match.group(1).strip()
                    win_counter[winner] += 1
                    model_names.add(winner)
                    winner_found = True
                else:
                    print(f"[WARN] Could not extract winner from {report_path}")
            else:
                print(f"[WARN] Missing verdict report in: {report_path}")

            # 2. Fallback: Infer best model by mean metric from metrics CSV
            if not winner_found and metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if "model" in df.columns:
                    # Compute average of primary metric across models
                    score_by_model = df.groupby("model")[metrics].mean().to_dict(orient="index")
                    best_model = max(score_by_model.items(), key=lambda kv: kv[1].get("accuracy", 0))[0]
                    win_counter[best_model] += 1
                    model_names.update(score_by_model.keys())
                else:
                    print(f"[WARN] No 'model' column found in {metrics_path}")
            elif not metrics_path.exists():
                print(f"[WARN] Missing metrics file: {metrics_path}")

            # 3. Collect average metrics for final summary
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if "model" in df.columns:
                    for model in df["model"].unique():
                        model_df = df[df["model"] == model]
                        model_names.add(model)
                        for metric in metrics:
                            if metric in model_df.columns:
                                metric_sums[model][metric] += model_df[metric].mean()
                                metric_counts[model][metric] += 1

        # 4. Compute averaged metrics per model
        model_metrics_avg = {}
        for model in model_names:
            model_metrics_avg[model] = {}
            for metric in metrics:
                count = metric_counts[model][metric]
                if count > 0:
                    avg = metric_sums[model][metric] / count
                    model_metrics_avg[model][metric] = round(avg, 4)
                else:
                    model_metrics_avg[model][metric] = "N/A"

        # 5. Decide final winner
        most_wins = win_counter.most_common()
        if not most_wins:
            print("[ERROR] No models evaluated or no valid metric files found.")
            return

        if len(most_wins) > 1 and most_wins[0][1] == most_wins[1][1]:
            best_model = max(model_metrics_avg.items(), key=lambda kv: kv[1].get("accuracy", 0))[0]
            tie_break = True
        else:
            best_model = most_wins[0][0]
            tie_break = False

        # 6. Build Markdown summary
        lines = [
            f"# 🧪 Final Evaluation Summary — Dataset: {dataset_name}",
            "## 🏁 Model Comparison Summary\n",
            "| Model | Wins | " + " | ".join([m.capitalize() for m in metrics]) + " |",
            "|-------|------|" + "----|" * len(metrics)
        ]
        for model in sorted(model_names):
            row = [model, win_counter.get(model, 0)]
            row += [model_metrics_avg[model][m] for m in metrics]
            lines.append("| " + " | ".join(map(str, row)) + " |")

        lines.append("\n## 🏆 Final Verdict")
        if tie_break:
            lines.append(f"Tie in wins. Based on higher average accuracy, **{best_model}** is the best model overall.")
        else:
            lines.append(f"**{best_model}** wins the majority of variant comparisons and is declared best overall.")

        out_path = base_dir / output_file
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[INFO] Final dataset-level winner report written to: {out_path}")


    

    def analyze(self):
        if self.final_report == 1:
            print(f"Evaluating final report...")
            self.generate_final_winner_report_with_metrics_auto(
                dataset_name=self.dataset,
                metrics=["accuracy","precision","recall","f1","roc_auc","log_loss"],
                results_root=self.get_all_datasets_dir()
            )    
            return
        print(f"Run MANOVA analysis now...")
        # Path to the CSV result this experiment generated
        result_file = self.get_dataset_result_csv()
        
        print(f"[INFO] Running MANOVA analysis on {result_file}")
        run_full_manova_analysis(result_file,output_dir=self.get_metrics_dir(),dataset_name=self.dataset)
