# models/topogat_selector.py
"""
Provides dynamic model selection for different TopoGAT variants.
Each variant implements a different fusion strategy for combining node and topological features.
"""
from models.registry import MODEL_REGISTRY

def get_topogat_model(input_dim, topo_dim, hidden_dim=64, num_classes=2, heads=1, variant="basic"):
    """
    Factory method to instantiate a TopoGAT variant.

    Args:
        input_dim (int): Dimension of node features.
        topo_dim (int): Dimension of topological features.
        hidden_dim (int): Hidden dimension for projections and GAT layers.
        num_classes (int): Number of output classes.
        heads (int): Number of attention heads in GATConv.
        variant (str): One of ['basic', 'concat', 'node_aware', 'gated', 'attn'].

    Returns:
        nn.Module: A TopoGAT model instance.
    """
    variant = variant.lower()
    if variant == "basic":
        variant = "concat"
    if variant not in MODEL_REGISTRY:
        raise ValueError(f"Unknown TopoGAT variant: {variant}")
    return MODEL_REGISTRY[variant](
        input_dim=input_dim,
        topo_dim=topo_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        heads=heads
    )


   
   