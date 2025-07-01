import networkx as nx
import numpy as np
import torch
import gudhi as gd
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)

class TopologyProcessor:
    def __init__(self, max_dim=1, topo_dim=8):
        self.max_dim = max_dim
        self.topo_dim = topo_dim
        logger.info(f"Initialized TopologyProcessor with max_dim={max_dim}, topo_dim={topo_dim}")
        self.homology_counts = Counter()
        self.total_graphs = 0

    def compute_persistence_diagram_graph(self, data, return_lifetimes=False):
        G = self._build_graph(data)
        if not nx.is_connected(G):
            logger.warning("Graph not connected. Extracting largest connected component.")
            G = self._get_largest_connected_subgraph(G)

        try:
            dist_matrix = self._compute_distance_matrix(G)
            if np.isinf(dist_matrix).any():
                logger.warning("Distance matrix contains inf. Returning zero topo vector.")
                if return_lifetimes:
                    return torch.zeros(self.topo_dim), []
                else:
                    return torch.zeros(self.topo_dim)
            return self._extract_topological_features(dist_matrix, return_lifetimes)
        except Exception as e:
            logger.error(f"[Topo] Failed for graph: {e}")
            if return_lifetimes:
                return torch.zeros(self.topo_dim), []
            else:
                return torch.zeros(self.topo_dim)

    def _build_graph(self, data):
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        G.add_edges_from(edge_index.T.tolist())
        return G

    def _get_largest_connected_subgraph(self, G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, node_mapping)

    def _compute_distance_matrix(self, G):
        dist = dict(nx.all_pairs_shortest_path_length(G))
        num_nodes = len(G)
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist_matrix[i, j] = dist.get(i, {}).get(j, np.inf)
        return dist_matrix

    def _extract_topological_features(self, dist_matrix, return_lifetimes=False):
        rips = gd.RipsComplex(distance_matrix=dist_matrix)
        st = rips.create_simplex_tree(max_dimension=self.max_dim)
        dgm = st.persistence()
        feats, lifetimes, counts = self.persistence_to_vector(dgm, return_lifetimes=True)
        self.total_graphs += 1
        self.homology_counts.update(counts)
        if return_lifetimes:
            return torch.tensor(feats, dtype=torch.float), lifetimes
        else:
            return torch.tensor(feats, dtype=torch.float)

    def persistence_to_vector(self, dgm, return_lifetimes=False):
        features = [death - birth for dim, (birth, death) in dgm
                    if dim in [0, 1] and not np.isinf(death)]
        h_counts = Counter(dim for dim, (birth, death) in dgm if dim in [0, 1] and not np.isinf(death))
        if not features:
            logger.warning("No persistent H0 or H1 features found in diagram.")
        top_k = sorted(features, reverse=True)[:self.topo_dim]
        padded = np.pad(top_k, (0, self.topo_dim - len(top_k)), 'constant')
        if return_lifetimes:
            return padded, features, h_counts
        else:
            return padded

    def summarize_statistics(self, output_path=None):
        if self.total_graphs == 0:
            logger.warning("No graphs processed for topology summary.")
            return
        summary = {f"H{dim}_avg_count": self.homology_counts[dim] / self.total_graphs for dim in self.homology_counts}
        logger.info("Topological Feature Summary: %s", summary)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v:.4f}\n")
        return summary
