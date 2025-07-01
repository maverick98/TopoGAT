# datasets/graph_dataset.py

import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from utils.topology import TopologyProcessor
import logging

logger = logging.getLogger(__name__)

class GraphTopoDataset:
    def __init__(self, name, topo_dim=8, seed=42):
        """
        Initializes the graph dataset with topological features.

        Args:
            name (str): Name of the TUDataset to load.
            topo_dim (int): Number of topological features to retain.
            seed (int): Random seed for reproducibility.
        """
        self.name = name
        self.seed = seed
        self.processor = TopologyProcessor(max_dim=1, topo_dim=topo_dim)

        raw_dataset = TUDataset(root=f"data/{name}", name=name)
        self._load_cached_or_compute_topo_features(raw_dataset)
        self._enrich_graphs_with_topo(raw_dataset)
        self.num_classes = raw_dataset.num_classes

    def _load_cached_or_compute_topo_features(self, raw_dataset):
        """
        Loads cached topological features if available, otherwise computes and caches them.

        Also saves a histogram of persistent H1 lifetimes.

        Args:
            raw_dataset (Dataset): Raw TUDataset graphs.
        """
        cache_file = f"data/{self.name}/ph_topo.pt"
        hist_path = f"data/{self.name}/ph_histogram.png"

        if os.path.exists(cache_file):
            self.topo_features = torch.load(cache_file)
            logger.info(f"[Topo] Loaded cached PH features from {cache_file}")
        else:
            self.topo_features, all_lifetimes = self._compute_and_cache_topo_features(raw_dataset)
            torch.save(self.topo_features, cache_file)
            logger.info(f"[Topo] Saved PH features to {cache_file}")

            if all_lifetimes:
                plt.hist(all_lifetimes, bins=30, color='skyblue', edgecolor='black')
                plt.title("PH H1 Lifetime Distribution")
                plt.xlabel("Lifetime")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(hist_path)
                logger.info(f"[Topo] Saved H1 lifetime histogram to {hist_path}")

        self.actual_topo_dim = self.topo_features.shape[1] + 1
        logger.info(f"[INFO] Detected topo vector dimension: {self.actual_topo_dim}")

    def _compute_and_cache_topo_features(self, raw_dataset):
        """
        Computes persistent homology vectors for each graph and caches them.

        Args:
            raw_dataset (Dataset): Raw TUDataset graphs.

        Returns:
            tuple: (Tensor of topo features, list of H1 lifetimes)
        """
        topo_features = []
        all_lifetimes = []
        for data in raw_dataset:
            vec, lifetimes = self.processor.compute_persistence_diagram_graph(data, return_lifetimes=True)
            topo_features.append(vec)
            all_lifetimes.extend(lifetimes)
        return torch.stack(topo_features), all_lifetimes

    def _enrich_graphs_with_topo(self, raw_dataset):
        """
        Attaches topological vectors and node degrees to each graph.

        Each graph's topological vector is repeated across its nodes
        and concatenated with node degree, enabling fusion in the model.

        Args:
            raw_dataset (Dataset): Raw TUDataset graphs.
        """
        self.graphs = []
        for i, data in enumerate(raw_dataset):
            topo_vec = self.topo_features[i]
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).unsqueeze(-1)
            enriched_topo = torch.cat([topo_vec.unsqueeze(0).repeat(data.num_nodes, 1), deg], dim=1)
            data.topo = enriched_topo
            self.graphs.append(data)

    def get_loaders(self, batch_size=32, seed=None):
        """
        Splits the dataset into train/test loaders.

        Args:
            batch_size (int): Batch size for the data loader.
            seed (int): Optional seed for reproducibility.

        Returns:
            tuple: (train_loader, test_loader)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        dataset = list(self.graphs)
        dataset = [d for d in dataset if hasattr(d, 'x') and hasattr(d, 'edge_index')]
        random.shuffle(dataset)
        split = int(0.8 * len(dataset))
        return DataLoader(dataset[:split], batch_size=batch_size), DataLoader(dataset[split:], batch_size=batch_size)
