import torch  # PyTorch core library for tensors and operations
import torch.nn.functional as F  # Functional API for layers and activations
from torch_geometric.nn import global_mean_pool, GATConv  # PyG GNN components: pooling and Graph Attention Convolution
import torch.nn as nn  # PyTorch neural network module
import logging  # Standard logging module
import os  # Filesystem operations
from abc import ABC, abstractmethod  # Abstract base classes for enforcing method implementation in subclasses
from models.registry import register_topogat  # Model registry system

logger = logging.getLogger(__name__)

class BaseTopoGAT(nn.Module, ABC):
    """
    Abstract Base Class for TopoGAT variants.
    Encapsulates shared logic for TopoGAT architectures, including projection layers,
    GAT convolutional layers, and classification head. Subclasses must define fusion strategy.
    """
    def __init__(self, input_dim, topo_dim, hidden_dim=64, num_classes=2, heads=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.topo_proj = nn.Linear(topo_dim, hidden_dim)  # Projects topological features to hidden_dim
        self.x_proj = nn.Linear(input_dim, hidden_dim)  # Projects node features to hidden_dim

        # Delay fused_dim and GAT layer initialization
        self.fused_dim = None
        self.gat1 = None
        self.gat2 = None

        self.classifier = nn.Linear(hidden_dim, num_classes)  # Final classification layer

    def finalize_setup(self):
        """
        Finalize model setup after subclass-specific layers are defined.
        Computes fused dimension and initializes GAT layers.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, self.hidden_dim)
            fused = self.fuse_features(dummy, dummy)
            self.fused_dim = fused.size(1)

        self.gat1 = GATConv(self.fused_dim, self.hidden_dim, heads=1, dropout=0.2)
        self.gat2 = GATConv(self.hidden_dim, self.hidden_dim, concat=False, dropout=0.2)

    @abstractmethod
    def fuse_features(self, x, topo):
        """
        Fuse projected node features and topological features.
        Must be implemented by subclasses.

        Args:
            x (Tensor): Projected node features. [N, d]
            topo (Tensor): Projected topo features. [N, d]

        Returns:
            Tensor: Fused features.
        """
        pass

    def forward(self, data):
        """
        Forward pass through TopoGAT model.

        Args:
            data: PyG Batch containing x, topo, edge_index, batch.

        Returns:
            Tensor: Log-softmax class probabilities. [num_graphs, num_classes]
        """
        x = F.relu(self.x_proj(data.x))  # Project and activate node features
        topo = F.relu(self.topo_proj(data.topo))  # Project and activate topological features
        h = self.fuse_features(x, topo)  # Apply subclass-specific fusion
        h = F.elu(self.gat1(h, data.edge_index))  # First GAT layer
        h = F.dropout(h, p=0.2, training=self.training)  # Dropout for regularization
        h = self.gat2(h, data.edge_index)  # Second GAT layer
        graph_embed = global_mean_pool(h, data.batch)  # Graph-level representation
        return F.log_softmax(self.classifier(graph_embed), dim=-1)  # Predict class probabilities

    def dump_internal_statistics(self, data, prefix="outputs/stat"):
        """
        If the model supports it, extract and save attention, gate, or transformer stats.
        """
        try:
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            x_proj = F.relu(self.x_proj(data.x))
            topo_proj = F.relu(self.topo_proj(data.topo))
            if hasattr(self, 'get_last_attention_weights'):
                attn = self.get_last_attention_weights(x_proj, topo_proj)
                torch.save(attn, f"{prefix}_attn.pt")
                logger.info(f"Saved attention weights to {prefix}_attn.pt")
            if hasattr(self, 'get_last_gate_values'):
                gate = self.get_last_gate_values(topo_proj)
                torch.save(gate, f"{prefix}_gates.pt")
                logger.info(f"Saved gate values to {prefix}_gates.pt")
            if hasattr(self, 'get_last_transformer_weights'):
                trans_attn = self.get_last_transformer_weights(x_proj, topo_proj)
                torch.save(trans_attn, f"{prefix}_transformer.pt")
                logger.info(f"Saved transformer attention to {prefix}_transformer.pt")
        except Exception as e:
            logger.warning(f"Failed to extract internal stats: {e}")

@register_topogat("concat")
class ConcatTopoGAT(BaseTopoGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finalize_setup()

    def fuse_features(self, x, topo):
        if x.shape != topo.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, topo {topo.shape}")
        return torch.cat([x, topo], dim=-1)

@register_topogat("node_aware")
class NodeAwareTopoGAT(BaseTopoGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finalize_setup()

    def fuse_features(self, x, topo):
        if x.shape != topo.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, topo {topo.shape}")
        return x * topo + x

@register_topogat("gated")
class GatedTopoGAT(BaseTopoGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.finalize_setup()

    def fuse_features(self, x, topo):
        if x.shape != topo.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, topo {topo.shape}")
        gate = torch.sigmoid(self.gate_layer(topo))
        return x * gate + topo * (1 - gate)

    def get_last_gate_values(self, topo):
        return torch.sigmoid(self.gate_layer(topo)).detach().cpu().numpy()

@register_topogat("attn")
class AttentionTopoGAT(BaseTopoGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        self.finalize_setup()

    def fuse_features(self, x, topo):
        if x.shape != topo.shape:
            raise ValueError(f"Shape mismatch: x {x.shape}, topo {topo.shape}")
        joint = torch.cat([x, topo], dim=-1)
        alpha = self.attn_layer(joint)
        return alpha * x + (1 - alpha) * topo

    def get_last_attention_weights(self, x, topo):
        joint = torch.cat([x, topo], dim=-1)
        return self.attn_layer(joint).detach().cpu().numpy()

@register_topogat("transformer")
class TopoTransformerGAT(BaseTopoGAT):
    """
    TopoGAT variant using transformer-style scaled dot-product attention for fusion.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_x = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_x = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_x = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.query_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_t = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fusion_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.finalize_setup()

    def scaled_dot_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights

    def fuse_features(self, x, topo):
        Q_x = self.query_x(x)
        K_x = self.key_x(x)
        V_x = self.value_x(x)
        out_x, self.x_weights = self.scaled_dot_attention(Q_x, K_x, V_x)

        Q_t = self.query_t(topo)
        K_t = self.key_t(topo)
        V_t = self.value_t(topo)
        out_t, self.t_weights = self.scaled_dot_attention(Q_t, K_t, V_t)

        fused = torch.cat([out_x, out_t], dim=-1)
        return self.fusion_proj(fused)

    def get_last_transformer_weights(self, x, topo):
        Q_x = self.query_x(x)
        K_x = self.key_x(x)
        _, x_weights = self.scaled_dot_attention(Q_x, K_x, self.value_x(x))

        Q_t = self.query_t(topo)
        K_t = self.key_t(topo)
        _, t_weights = self.scaled_dot_attention(Q_t, K_t, self.value_t(topo))

        return {
            "x_weights": x_weights.detach().cpu().numpy(),
            "topo_weights": t_weights.detach().cpu().numpy()
        }
