# tests/test_topogat_variants_rigorous.py
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch.optim import Adam
from models.topogat import (
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
)

@pytest.fixture(params=[1, 3])
def dummy_data_batch(request):
    batch_size = request.param
    num_nodes = 5 * batch_size
    x = torch.rand(num_nodes, 10)          # input_dim=10
    topo = torch.rand(num_nodes, 6)        # topo_dim=6

    # Connect nodes sequentially within each graph in the batch
    edge_indices = []
    for b in range(batch_size):
        start = b * 5
        edges = torch.tensor([
            [start, start + 1, start + 2, start + 3],
            [start + 1, start + 2, start + 3, start + 4]
        ], dtype=torch.long)
        edge_indices.append(edges)
    edge_index = torch.cat(edge_indices, dim=1)

    batch = torch.zeros(num_nodes, dtype=torch.long)
    for b in range(batch_size):
        batch[b*5:(b+1)*5] = b

    return Data(x=x, topo=topo, edge_index=edge_index, batch=batch)

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
])
def test_forward_and_backward(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=4)
    model.train()

    out = model(dummy_data_batch)
    batch_size = dummy_data_batch.batch.max().item() + 1

    # Shape check
    assert out.shape == (batch_size, 4), f"Output shape mismatch for {ModelClass.__name__}"

    # Backward check
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), f"Missing grads for {ModelClass.__name__}"

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
])
def test_training_step(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=4)
    optimizer = Adam(model.parameters(), lr=0.01)
    model.train()

    out = model(dummy_data_batch)
    target = torch.randint(0, 4, (dummy_data_batch.batch.max().item() + 1,))
    loss_fn = torch.nn.NLLLoss()
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
])
def test_dropout_behavior(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=4)
    model.train()
    out_train = model(dummy_data_batch)

    model.eval()
    out_eval = model(dummy_data_batch)

    assert out_train.shape == out_eval.shape, f"Output shape mismatch for {ModelClass.__name__}"
    # Outputs in train and eval mode should differ due to dropout randomness
    assert not torch.allclose(out_train, out_eval), f"Dropout inactive for {ModelClass.__name__}"

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGAT,
    NodeAwareTopoGAT,
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
])
def test_invalid_input_dims(ModelClass):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=4)

    data = Data(
        x=torch.rand(5, 8),  # Wrong input_dim (8 instead of 10)
        topo=torch.rand(5, 6),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.zeros(5, dtype=torch.long)
    )
    with pytest.raises(RuntimeError):
        model(data)

@pytest.mark.parametrize("ModelClass", [
    GatedTopoGAT,
    AttentionTopoGAT,
    TopoTransformerGAT,
])
def test_internal_stats_extraction(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=4)
    model.eval()

    # Try dump_internal_statistics without errors
    model.dump_internal_statistics(dummy_data_batch, prefix="tests/stat")

    # Check if relevant internal getter methods exist and output as expected
    if hasattr(model, "get_last_gate_values"):
        gates = model.get_last_gate_values(dummy_data_batch.topo)
        assert isinstance(gates, (list, tuple, torch.Tensor, np.ndarray)), "Gates output type invalid"

    if hasattr(model, "get_last_attention_weights"):
        attn = model.get_last_attention_weights(
            F.relu(model.x_proj(dummy_data_batch.x)),
            F.relu(model.topo_proj(dummy_data_batch.topo))
        )
        assert attn is not None, "Attention weights should not be None"

    if hasattr(model, "get_last_transformer_weights"):
        trans_attn = model.get_last_transformer_weights(
            F.relu(model.x_proj(dummy_data_batch.x)),
            F.relu(model.topo_proj(dummy_data_batch.topo))
        )
        assert isinstance(trans_attn, dict), "Transformer attention output must be a dict"
