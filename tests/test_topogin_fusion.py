import torch
from models.topogin import (
    ConcatTopoGIN,
    NodeAwareTopoGIN,
    GatedTopoGIN,
    AttentionTopoGIN,
    TransformerTopoGIN
)

def test_concat_fusion():
    model = ConcatTopoGIN(input_dim=4, topo_dim=4)
    x = torch.randn(10, 64)
    topo = torch.randn(10, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == (10, 128)

def test_node_aware_fusion():
    model = NodeAwareTopoGIN(input_dim=4, topo_dim=4)
    x = torch.ones(5, 64)
    topo = torch.ones(5, 64) * 0.5
    fused = model.fuse_features(x, topo)
    expected = x * topo + x
    assert torch.allclose(fused, expected)

def test_gated_fusion():
    model = GatedTopoGIN(input_dim=4, topo_dim=4)
    x = torch.randn(3, 64)
    topo = torch.randn(3, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == x.shape

def test_attention_fusion():
    model = AttentionTopoGIN(input_dim=4, topo_dim=4)
    x = torch.randn(2, 64)
    topo = torch.randn(2, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == x.shape

def test_transformer_fusion():
    model = TransformerTopoGIN(input_dim=4, topo_dim=4)
    x = torch.randn(2, 64)
    topo = torch.randn(2, 64)
    fused = model.fuse_features(x, topo)
    assert fused.shape == x.shape
