# datasets/graph_dataset.py

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from utils.topology import TopologyProcessor
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphTopoDataset:
    def __init__(self, name, topo_dim=8, seed=42):
        """
        Initializes the graph dataset with persistent homology features.

        Args:
            name (str): Name of the TUDataset to load.
            topo_dim (int): Number of topological features to retain.
            seed (int): Random seed for reproducibility.
        """
        self.name = name
        self.seed = seed
        self.processor = TopologyProcessor(max_dim=1, topo_dim=topo_dim)

        dataset = TUDataset(root=f"data/{name}", name=name)
        self._load_or_compute_topo_features(dataset)
        self._attach_topo_and_degree(dataset)
        self.num_classes = dataset.num_classes

    def _load_or_compute_topo_features(self, dataset):
        cache_path = f"data/{self.name}/ph_topo.pt"
        hist_path = f"data/{self.name}/ph_histogram.png"

        if os.path.exists(cache_path):
            self.topo_features = torch.load(cache_path)
            logger.info(f"Loaded cached PH features from {cache_path}")
        else:
            self.topo_features, all_lifetimes = self._compute_topo_features(dataset)
            torch.save(self.topo_features, cache_path)
            logger.info(f"Saved PH features to {cache_path}")

            if all_lifetimes:
                plt.hist(all_lifetimes, bins=30, color='skyblue', edgecolor='black')
                plt.title("PH H1 Lifetime Distribution")
                plt.xlabel("Lifetime")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(hist_path)
                logger.info(f"Saved PH lifetime histogram to {hist_path}")

        self.actual_topo_dim = self.topo_features.shape[1] + 1  # +1 for degree
        logger.info(f"Detected topo vector dimension: {self.actual_topo_dim}")

    def _compute_topo_features(self, dataset):
        topo_list, all_lifetimes = [], []
        for data in dataset:
            vec, lifetimes = self.processor.compute_persistence_diagram_graph(data, return_lifetimes=True)
            topo_list.append(vec)
            all_lifetimes.extend(lifetimes)
        return torch.stack(topo_list), all_lifetimes

    def _attach_topo_and_degree(self, dataset):
        self.graphs = []
        for i, data in enumerate(dataset):
            topo_vec = self.topo_features[i]
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).unsqueeze(1)
            topo_per_node = topo_vec.unsqueeze(0).repeat(data.num_nodes, 1)
            data.topo = torch.cat([topo_per_node, deg], dim=1)
            self.graphs.append(data)

    def get_loaders(self, batch_size=32, seed=None):
        """
        Returns train and test loaders after 80/20 split.

        Args:
            batch_size (int): Batch size for DataLoader.
            seed (int, optional): Seed for reproducibility.

        Returns:
            tuple: (train_loader, test_loader)
        """
        seed = seed or self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dataset = [g for g in self.graphs if hasattr(g, 'x') and hasattr(g, 'edge_index')]
        random.shuffle(dataset)
        split_idx = int(0.8 * len(dataset))
        return (
            DataLoader(dataset[:split_idx], batch_size=batch_size),
            DataLoader(dataset[split_idx:], batch_size=batch_size)
        )
