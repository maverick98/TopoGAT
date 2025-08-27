import argparse
from models.gin import BaseGIN
from models.topogin_selector import get_topogin_model
from experiments.base_experiment import BaseTopoComparisonExperiment
from utils.config import load_config
class TopoGINvsGINExperiment(BaseTopoComparisonExperiment):
    def __init__(self, config,final_report,force_train_eval,dataset, variant="topogin-concat", seeds=20, use_wandb=False, output_dir="results/topogin_vs_gin"):
        # Save experiment metadata
        self.config=config
        self.final_report=final_report
        self.force_train_eval=force_train_eval
        self.dataset=dataset
        self.variant = variant
        self.seeds = seeds
        self.use_wandb = use_wandb
        self.base_model_name = "GIN"
        self.output_dir=output_dir
       

        # Initialize base comparison experiment with all core hooks
        super().__init__(
            config=config,
            final_report=self.final_report,
            force_train_eval=self.force_train_eval,
            topo_model='topogin',
            variant=None if self.final_report == 1 else self.variant,
            base_model_name=self.base_model_name,
            seeds=self.seeds,
            base_model_fn=None if self.final_report == 1 else BaseGIN,
            topo_model_fn =None if self.final_report == 1 else lambda input_dim, topo_dim, num_classes: get_topogin_model(
                            input_dim=input_dim,
                            topo_dim=topo_dim,
                            num_classes=num_classes,
                            variant=self.variant
                ),
            output_dir=self.output_dir
           
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--final_report', type=str, default=0, choices=[0, 1], help='final report')
    parser.add_argument('--force_train_eval', type=int, default=0, choices=[0, 1], help='force_train_eval')
    parser.add_argument('--variant', type=str, default='topogin_concat', help='Topo variant (e.g., basic, lap)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for statistical significance')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')

    args = parser.parse_args()
    config = load_config()
    config["dataset"] = args.dataset  
    config["topo_dim"]=config["topo_dim"]-1

    experiment = TopoGINvsGINExperiment(
        config=config,
        final_report=args.final_report,
        force_train_eval=args.force_train_eval,
        dataset=args.dataset,
        variant=args.variant,
        seeds=args.runs,
        use_wandb=args.use_wandb,
        output_dir= f"topogin/{args.dataset}" if args.final_report == 1 else f"topogin/{args.dataset}/{args.variant}"
    )
    experiment.run()
