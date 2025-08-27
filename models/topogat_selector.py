# models/topogat_selector.py

import inspect
from models.registry import TOPOGAT_REGISTRY


def get_topogat_model(input_dim, topo_dim, hidden_dim=64, num_classes=2, heads=1, variant="basic"):

    variant = variant.lower()
    if variant == "topogat-basic":
        variant = "topogat-concat"

    if variant not in TOPOGAT_REGISTRY:
        raise ValueError(f"[ERROR] Unknown TopoGAT variant: '{variant}'. Available: {list(TOPOGAT_REGISTRY.keys())}")

    variant = variant.lower()
    if variant == "basic":
        variant = "concat"
    if variant not in TOPOGAT_REGISTRY:
        raise ValueError(f"Unknown TopoGAT variant: {variant}")
    return TOPOGAT_REGISTRY[variant](
        input_dim=input_dim,
        topo_dim=topo_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_classes=num_classes  
    )
