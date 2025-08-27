#trainers/graph_trainer.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import random
import logging
from datasets.graph_dataset import GraphTopoDataset

logger = logging.getLogger(__name__)

class GraphTrainer:
    def __init__(self, config, seed=42):
        print(f"[INFO ] config = {config}")
        self.seed = seed
        self.config = config
        self.dataset_name = config['dataset']
        self.dataset = GraphTopoDataset(self.dataset_name, topo_dim=config['topo_dim'], seed=seed)
        self.train_loader, self.test_loader = self.dataset.get_loaders(seed=seed)
        self.sample = self.dataset.graphs[0]
        if not hasattr(self.sample, 'topo'):
            self.sample.topo = self.dataset.topo_features[0]
        print(f'actual_topo_dim {self.dataset.actual_topo_dim}')    
        self.topo_dim = self.dataset.actual_topo_dim


    def train_and_eval_model(self, model, label):
        logger.info(f"\n[Training {label}]")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        for epoch in range(1, self.config['epochs'] + 1):

            loss = self.train_one_epoch(model, optimizer)
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")

        acc = self.evaluate(model)
        self.log_metrics(label, acc)
        return acc

    def train_one_epoch(self, model, optimizer):
        model.train()
        total_loss = 0
        for batch in self.train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self, model):
        model.eval()
        correct = 0
        total = 0
        all_internal_stats = []

        for batch in self.test_loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)

            # Optional internal statistics dump if model supports it
            if hasattr(model, 'dump_internal_statistics'):
                try:
                    stats = model.dump_internal_statistics(batch)
                    all_internal_stats.append(stats)
                except Exception as e:
                    logger.warning(f"Failed to collect internal statistics: {e}")

        acc = correct / total
        if all_internal_stats:
            logger.info("Internal model statistics collected across batches.")
            # Could extend to save to file, visualize, or aggregate here
        return acc

    def log_metrics(self, label, acc):
        logger.info(f"{label} Test Accuracy: {acc:.4f}")
