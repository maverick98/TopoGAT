import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
'''
    Importing PyTorch essentials and the GATConv layer from torch_geometric, 
    which implements the Graph Attention Network (GAT) convolution.
'''

class TopoGAT(torch.nn.Module):
    """
    TopoGAT: Graph Attention Network enhanced with topological features using Persistent Homology.

    This model extends the standard GAT by concatenating node features with topological embeddings,
    enabling the network to capture both feature-level and topology-aware patterns in graph data.

    Args:
        in_channels (int): Number of input features per node (including topological dimensions).
        out_channels (int): Number of output classes (for classification).
        topo_dim (int): Dimension of the topological feature vector to be concatenated.
    """
    def __init__(self, in_channels, out_channels, topo_dim=8):
        super(TopoGAT, self).__init__()
        
        # First GAT layer:
        # - Takes the concatenated feature vector of each node (original + topological).
        # - Outputs 8-dimensional features per head with 8 heads.
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        
        # Second GAT layer:
        # - Input is 8 * 8 = 64-dimensional (from all heads).
        # - Outputs class scores (logits) directly.
        # - heads=1 and concat=False means outputs are not concatenated, but averaged.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        """
        Performs a forward pass of the TopoGAT model.

        Args:
            data (torch_geometric.data.Data): Graph data containing:
                - data.x: Node feature matrix.
                - data.edge_index: Graph connectivity in COO format.
                - data.topo: Topological feature matrix.

        Returns:
            torch.Tensor: Log probabilities for each class (for each node).
        """
        # Concatenate original node features with persistent homology features
        x = torch.cat([data.x, data.topo], dim=1)
        # Apply dropout to the input features (regularization)
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply first GATConv layer with ELU activation
        x = F.elu(self.conv1(x, data.edge_index))
        # Dropout again before final layer (prevents overfitting)
        x = F.dropout(x, p=0.6, training=self.training)
        # Final GATConv layer + log_softmax for classification
        return F.log_softmax(self.conv2(x, data.edge_index), dim=1)
        '''
            Summary

                This module is a 2-layer Graph Attention Network, but augmented by concatenating topological features 
                (data.topo) to node embeddings.

                The key novelty lies in injecting persistent homology-derived features 
                before feeding them into attention layers — enhancing the model's ability to 
                capture higher-order structural information.
        '''