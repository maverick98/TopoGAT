import torch
from models.topogat import (
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT
)


def test_concat_fusion():
    model = ConcatTopoGAT(input_dim=64, topo_dim=64)
    x = torch.randn(10, 64)
    topo = torch.randn(10, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == (10, 128), f"Expected shape (10,128), got {fused.shape}"


def test_node_aware_fusion():
    model = NodeAwareTopoGAT(input_dim=64, topo_dim=64)
    x = torch.ones(5, 64)
    topo = torch.ones(5, 64) * 0.5
    fused = model.fuse_features(x, topo)
    expected = x * topo + x
    assert torch.allclose(fused, expected, atol=1e-5), "Node-aware fusion output mismatch."


def test_gated_fusion():
    model = GatedTopoGAT(input_dim=64, topo_dim=64)
    x = torch.randn(3, 64)
    topo = torch.randn(3, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == x.shape, f"Gated fusion output shape mismatch: {fused.shape}"


def test_attention_fusion():
    model = AttentionTopoGAT(input_dim=64, topo_dim=64)
    x = torch.randn(2, 64)
    topo = torch.randn(2, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == x.shape, f"Attention fusion output shape mismatch: {fused.shape}"
