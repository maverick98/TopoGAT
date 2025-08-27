#experiments/topogat_vs_gat.py
import argparse
from models.gat import BaseGAT
from models.topogat_selector import get_topogat_model
from experiments.base_experiment import BaseTopoComparisonExperiment
from utils.config import load_config
class TopoGATvsGATExperiment(BaseTopoComparisonExperiment):
    def __init__(self, config,final_report,force_train_eval,dataset, variant, seeds, use_wandb, output_dir):
        # Save experiment metadata
        self.config=config
        self.final_report=final_report
        self.force_train_eval=force_train_eval
        self.dataset=dataset
        self.variant = variant
        self.seeds = seeds
        self.use_wandb = use_wandb
        self.base_model_name = "GAT"
        self.output_dir=output_dir
       

        # Initialize base comparison experiment with all core hooks
        super().__init__(
            config=config,
            final_report=self.final_report,
            force_train_eval=self.force_train_eval,
            topo_model='topogat',
            variant=None if self.final_report == 1 else self.variant,
            base_model_name=self.base_model_name,
            seeds=self.seeds,
            base_model_fn=None if self.final_report == 1 else BaseGAT,
            topo_model_fn =None if self.final_report == 1 else lambda input_dim, topo_dim, num_classes: get_topogat_model(
                            input_dim=input_dim,
                            topo_dim=topo_dim,
                            num_classes=num_classes,
                            variant=self.variant
                ),
            output_dir=self.output_dir
        )

        #self.setup_experiment(output_dir, logger_name="TopoGATvsGAT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--final_report', type=int, default=0, choices=[0, 1], help='final report')
    parser.add_argument('--force_train_eval', type=int, default=0, choices=[0, 1], help='force_train_eval')
    parser.add_argument('--variant', type=str, default='basic', help='Topo variant (e.g., basic, lap)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for statistical significance')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    config["dataset"] = args.dataset  # Inject dataset name into config
    if args.dataset == 'ENZYMES':
       print(f'if dataset is ENZYMES configs num_classes should be 6') 
       config["num_classes"]=6
       print(f'if dataset is ENZYMES {config}') 

    # Initialize and run the experiment
    experiment = TopoGATvsGATExperiment(
        config=config,
        final_report=args.final_report,
        force_train_eval=args.force_train_eval,
        dataset=args.dataset,
        variant=args.variant,
        seeds=args.runs,
        use_wandb=args.use_wandb,
        output_dir= f"topogat/{args.dataset}" if args.final_report == 1 else f"topogat/{args.dataset}/{args.variant}"
    )

    # This invokes the training and evaluation pipeline
    experiment.run()
