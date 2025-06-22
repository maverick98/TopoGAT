from trainers.trainer import Trainer
from models.topogat import TopoGAT
from data.dataset import CoraDataset
from evaluators.evaluator import Evaluator
import torch

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

    data, in_channels, out_channels = CoraDataset().get_data()
    model = TopoGAT(in_channels, out_channels)
    model.load_state_dict(torch.load('model.pt'))
    evaluator = Evaluator(model, data)
    acc = evaluator.evaluate()
    print(f"Test Accuracy: {acc:.4f}")