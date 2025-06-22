import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from utils.topology import TopologyProcessor

class CoraDataset:
    def __init__(self, name="Cora"):
        if name in ["Cora", "PubMed", "Citeseer"]:
            self.dataset = Planetoid(root=f'data/{name}', name=name, transform=NormalizeFeatures())
        else:
            self.dataset = TUDataset(root=f'data/{name}', name=name)
        self.processor = TopologyProcessor()

    def get_data(self):
        data = self.dataset[0]
        data.topo = self.processor.compute_persistence_diagrams(data)
        return data, data.num_node_features + data.topo.shape[1], self.dataset.num_classes