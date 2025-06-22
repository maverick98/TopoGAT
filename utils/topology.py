import gudhi as gd
import numpy as np
import torch
import networkx as nx

class TopologyProcessor:
    def __init__(self, max_dim=1, topo_dim=8):
        self.max_dim = max_dim
        self.topo_dim = topo_dim

    def compute_persistence_diagrams(self, data):
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        G.add_edges_from(edge_index.T.tolist())
        diagrams = []
        for node in range(data.num_nodes):
            neighbors = list(G.neighbors(node)) + [node]
            subgraph = G.subgraph(neighbors)
            rips = gd.RipsComplex(points=np.array([[i] for i in subgraph.nodes]), max_edge_length=2.0)
            simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dim)
            diag = simplex_tree.persistence()
            pers_feats = self.persistence_to_vector(diag)
            diagrams.append(pers_feats)
        return torch.tensor(diagrams, dtype=torch.float)

    def persistence_to_vector(self, diag):
        features = [birth-death for dim, (birth, death) in diag if dim == 1]
        top_k = sorted(features, reverse=True)[:self.topo_dim]
        return np.pad(top_k, (0, self.topo_dim - len(top_k)), 'constant')