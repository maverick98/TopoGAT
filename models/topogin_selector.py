# models/topogin_selector.py
"""
Provides dynamic model selection for different TopoGIN variants.
Each variant implements a different fusion strategy for combining node and topological features.
"""

from models.registry import TOPOGIN_REGISTRY
import inspect





def get_topogin_model(input_dim, topo_dim, hidden_dim=64, num_classes=2, heads=1, variant="basic"):

    variant = variant.lower()
    if variant == "basic":
        variant = "topogin-gatedmlp"

    if variant not in TOPOGIN_REGISTRY:
        raise ValueError(f"[ERROR] Unknown TopoGIN variant: '{variant}'. Available: {list(TOPOGIN_REGISTRY.keys())}")

    variant = variant.lower()
    if variant == "basic":
        variant = "topogin-gatedmlp"
    if variant not in TOPOGIN_REGISTRY:
        raise ValueError(f"Unknown TopoGIN variant: {variant}")
    return TOPOGIN_REGISTRY[variant](
        input_dim=input_dim,
        topo_dim=topo_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_classes=num_classes  
        )
