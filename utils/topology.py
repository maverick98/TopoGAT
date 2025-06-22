import gudhi as gd
import numpy as np
import torch
import networkx as nx

class TopologyProcessor:
    def __init__(self, max_dim=1, topo_dim=8):
        ''' 
            max_dim: The maximum homological dimension to compute 
            (1 means we are computing connected components and loops). 
        '''
        self.max_dim = max_dim 
        '''
            topo_dim: The size of the topological feature vector per node (e.g., top-8 persistent features).
        '''
        self.topo_dim = topo_dim

    def compute_persistence_diagrams(self, data):
        """
            
            Computes node-level persistent homology features for a given graph.

            For each node in the graph, this method constructs its 1-hop neighborhood subgraph,
            computes the persistent homology using a Rips complex on the node indices (as proxy coordinates),
            and transforms the persistence diagram into a fixed-size vector using topological feature lifetimes.

            Parameters
            ----------
            data : torch_geometric.data.Data
                A graph data object from PyTorch Geometric. Expected to have 'edge_index' and 'num_nodes'.

            Returns
            -------
            torch.Tensor
                A tensor of shape [num_nodes, topo_dim], where each row is the topological feature
                vector for a node, derived from its 1-hop neighborhood.

            Notes
            -----
            - The persistence diagram is computed for dimension 1 (loops).
            - Node indices are used as coordinates for the Rips complex.
            - This is a structural/topological approximation and not geometric.
            - The output tensor can be concatenated to original node features for downstream tasks.

        """
        G = nx.Graph()
        edge_index = data.edge_index.numpy()
        G.add_edges_from(edge_index.T.tolist())
        '''
            Converts the PyTorch edge_index (standard format in torch_geometric) to a networkx graph.

            edge_index.T.tolist() gives  edge pairs in list form like [(src1, dst1), (src2, dst2), ...].

        '''

        diagrams = []
        for node in range(data.num_nodes):
            '''
                Let's store one persistence diagram per node.
            '''
            neighbors = list(G.neighbors(node)) + [node]
            subgraph = G.subgraph(neighbors)
            '''
                Get all 1-hop neighbors of a node (its immediate neighborhood).
                Include the node itself.
                Extract the induced subgraph from these nodes.
            '''    



            '''
                The Vietoris–Rips Complex is a type of simplicial complex built from a set of points, 
                where a k-simplex (like an edge, triangle, tetrahedron, etc.) is formed 
                whenever all its (k+1) vertices are pairwise within a specified distance.
            '''
            '''
                Let's use Vietoris–Rips Complex for now due to below reason.
                It's computationally efficient.
                Well-supported in gudhi.
                Provides a reasonable topological approximation from local graph neighborhoods.
            '''
            rips = gd.RipsComplex(points=np.array([[i] for i in subgraph.nodes]), max_edge_length=2.0)
            simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dim)
            '''
                We are embedding each node ID as a 1D point (e.g., node 5 becomes point [5]).
                This may not capture true geometry — but as a placeholder, 
                it builds a Rips complex over "indices-as-points".
                RipsComplex creates a simplicial complex from these distances.
                Then create_simplex_tree builds a simplex tree up to max_dim.
            '''



            diag = simplex_tree.persistence()
            '''
                Now we get birth and death times of topological features (e.g., loops in H1).
            '''
            pers_feats = self.persistence_to_vector(diag)
            diagrams.append(pers_feats)
            '''
                Convert the diagram to a vector (next method).
                Append it to the list of topological features (1 vector per node).
            '''
        return torch.tensor(diagrams, dtype=torch.float)
        '''
            Return all vectors as a PyTorch tensor → shape will be [num_nodes, topo_dim].
        '''

    def persistence_to_vector(self, diag):
        """
            Converts a persistence diagram into a fixed-size topological feature vector.

            Parameters
            ----------
            diag : list of tuples
                The persistence diagram returned by GUDHI's simplex_tree.persistence(). 
                Each entry is a tuple (dim, (birth, death)) representing a topological feature 
                of a given dimension that appears at 'birth' and disappears at 'death'.

            Returns
            -------
            np.ndarray
                A 1D numpy array of length `topo_dim`, containing the top-k largest lifetimes 
                (death - birth) of 1-dimensional topological features (H1 - loops). 
                If there are fewer than `topo_dim` such features, the remaining entries are zero-padded.
            
            Notes
            -----
            - This function focuses only on H1 features (loops).
            - It extracts their lifetimes and encodes them as a vector of fixed dimension.
            - This vector can be concatenated to other node features for downstream learning.
        """
        
        features = [birth-death for dim, (birth, death) in diag if dim == 1]
        '''
            This extracts loop features (H1):
            It computes lifetimes (death - birth) for each persistent 1D feature (loops).
            We are ignoring H0 (connected components) and higher dimensions.
        '''
        top_k = sorted(features, reverse=True)[:self.topo_dim]
        '''
            Select top-k longest-living features → These are most topologically significant.
        '''
        return np.pad(top_k, (0, self.topo_dim - len(top_k)), 'constant')
        '''
            Pad with zeros if there are fewer than topo_dim features.
            Ensures every vector has the same length.
        '''

'''
    Here's the rationale for focusing on H₁ (1-dimensional holes/loops) and ignoring H₀ (connected components) and 
    higher-dimensional features (H₂ and beyond):
    Why We Focus on H₁ (Loops)

    Loops are key to capturing higher-order structure:

        GNNs (like GCN, GAT) already capture H₀ implicitly because they rely on message passing, 
        which naturally encodes connectivity (i.e., whether two nodes are in the same component).

        But loops (H₁) capture non-trivial circular dependencies, which are not easily detected by message passing alone. 
        They help differentiate between densely connected neighborhoods and more cyclic topologies.

    Higher-order structure boosts expressivity:

        Graph isomorphism tests (like Weisfeiler-Lehman) fail in graphs with similar local structure 
        but different global topology (e.g., cycles vs trees). H₁ can help disambiguate these.

    Empirical evidence:

        Multiple studies (e.g., TOGL, TopoGraph, PH-GCN) found H₁ features to be most useful and 
        stable for downstream learning tasks.

    Why We Ignore H₀ (Connected Components)

    Redundancy with node degree/connectivity:

        If a node is in a component with its neighbors, that info is already in the graph structure and 
        encoded in GAT/GNN layers.

        Including H₀ often just duplicates information that's already well captured by node embeddings.

    Less discriminative:

        All nodes in a connected graph will just yield a trivial H₀ diagram — with limited utility. 
        It won't help differentiate local topologies meaningfully.

    Why We Ignore H₂+ (Voids and beyond)

    Computational burden:

        Computing persistent homology in H₂ or higher dimensions on even moderate-sized graphs is computationally 
        very expensive.

        For large graphs like Cora or PubMed, the simplex growth is exponential and not scalable.

    Sparse or non-existent features:

        In most real-world graphs (especially citation/social/knowledge graphs), 
        H₂ features are rare or not meaningful unless you're in a spatial or 3D context (like molecules or meshes).

    Noise sensitivity:

        Higher-dimensional features tend to be unstable under small perturbations (less persistence), 
        especially when built from artificial coordinate embeddings (like using node IDs).

    Summary
    Homology Group | Captures	                        | Use in GNNs	               | Reason to Include
        H₀	       | Connected components               | Already encoded in GNNs	   | ❌ Mostly redundant
        H₁	       | Cycles / loops	                    |Rarely captured	           | ✅ Crucial for expressivity
        H₂+	       | Voids / cavities / higher-topology | Computationally hard, sparse | ❌ Expensive & often useless

'''


'''
     1. Čech Complex (Čech Complex)
Definition:

Given a set of points X in a metric space and a radius ϵ, the Čech complex includes a k-simplex 
whenever the (k+1)(k+1) balls of radius ϵϵ centered at the points have a non-empty common intersection.
Key Property:

    The Čech complex accurately recovers the underlying topology of the data (nerve theorem).

    It is guaranteed to be homotopy equivalent to the union of the balls.

Drawbacks:

    Hard to compute in practice because checking common intersections among multiple balls is computationally expensive.

    Not often used in practice for graphs or large datasets.

 2. Vietoris–Rips Complex (Used in RipsComplex)
Definition:

Includes a k-simplex whenever all pairs of (k+1) points are within distance ϵ.
Key Property:

    It is a computationally simpler approximation of the Čech complex.

    Easier to construct because it only checks pairwise distances.

Relationship:

    The Vietoris–Rips complex approximates the Čech complex.

    It overestimates the Čech complex (i.e., it may include more simplices).

    Still captures the important topological features, especially for small radii.

Why Rips is Preferred in Practice in TopoGAT:
    |Feature	|Čech Complex	|Vietoris–Rips Complex|
    |Theoretical Accuracy	|High (homotopy equivalent)	|Approximate|
    |Computational Cost	|High	|Moderate (pairwise distances)|
    |Library Support	|Limited|	Excellent (GUDHI, Ripser, etc.)|
    |Common Use in TDA	|Rare|	Standard|
     Other Complexes We might encounter:

    Alpha Complex: Built from Delaunay triangulation; closer to Čech but more efficient.

    Witness Complex: Useful when working with landmark points.

    Clique Complex: Used for graph filtrations (common in GNNs).

    Flag Complex: Synonym for Vietoris–Rips in the graph-theoretical setting.


'''


