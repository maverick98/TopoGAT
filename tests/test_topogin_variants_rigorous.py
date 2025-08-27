# tests/test_topogin_variants_rigorous.py
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch.optim import Adam
from models.topogin import (
    ConcatTopoGIN, NodeAwareTopoGIN, GatedTopoGIN, AttentionTopoGIN, TransformerTopoGIN
)

@pytest.fixture(params=[1, 4])
def dummy_data_batch(request):
    batch_size = request.param
    num_nodes = 5 * batch_size
    x = torch.rand(num_nodes, 10)          # input_dim=10
    topo = torch.rand(num_nodes, 6)        # topo_dim=6

    # Create edges by connecting nodes sequentially within each graph in the batch
    edge_indices = []
    for b in range(batch_size):
        start = b * 5
        # Simple chain graph: 0->1->2->3->4 per graph
        edges = torch.tensor([
            [start, start+1, start+2, start+3],
            [start+1, start+2, start+3, start+4]
        ], dtype=torch.long)
        edge_indices.append(edges)
    edge_index = torch.cat(edge_indices, dim=1)

    batch = torch.zeros(num_nodes, dtype=torch.long)
    for b in range(batch_size):
        batch[b*5:(b+1)*5] = b

    return Data(x=x, topo=topo, edge_index=edge_index, batch=batch)

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGIN,
    NodeAwareTopoGIN,
    GatedTopoGIN,
    AttentionTopoGIN,
    TransformerTopoGIN,
])
def test_forward_and_backward(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=3, dropout=0.1)
    model.train()

    out = model(dummy_data_batch)
    batch_size = dummy_data_batch.batch.max().item() + 1

    # Shape check
    assert out.shape == (batch_size, 3), f"Output shape mismatch for {ModelClass.__name__}"

    # Check backward pass works
    loss = out.sum()
    loss.backward()

    # Check gradients are populated for parameters
    grad_params = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grad_params), f"Some gradients missing for {ModelClass.__name__}"

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGIN,
    NodeAwareTopoGIN,
    GatedTopoGIN,
    AttentionTopoGIN,
    TransformerTopoGIN,
])
def test_training_step(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=3, dropout=0.1)
    optimizer = Adam(model.parameters(), lr=0.01)
    model.train()

    out = model(dummy_data_batch)
    target = torch.randint(0, 3, (dummy_data_batch.batch.max().item() + 1,))  # random target
    loss_fn = torch.nn.NLLLoss()
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # If we reach here, training step passed without error

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGIN,
    NodeAwareTopoGIN,
    GatedTopoGIN,
    AttentionTopoGIN,
    TransformerTopoGIN,
])
def test_dropout_behavior(ModelClass, dummy_data_batch):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=3, dropout=0.5)

    model.train()
    out_train = model(dummy_data_batch)

    model.eval()
    out_eval = model(dummy_data_batch)

    # Outputs should differ because of dropout randomness in training mode
    # But should be same shape and not equal exactly
    assert out_train.shape == out_eval.shape, f"Output shape mismatch for {ModelClass.__name__}"
    assert not torch.allclose(out_train, out_eval), f"Dropout seems inactive in {ModelClass.__name__}"

@pytest.mark.parametrize("ModelClass", [
    ConcatTopoGIN,
    NodeAwareTopoGIN,
    GatedTopoGIN,
    AttentionTopoGIN,
    TransformerTopoGIN,
])
def test_invalid_input_shapes(ModelClass):
    model = ModelClass(input_dim=10, topo_dim=6, hidden_dim=16, num_classes=3, dropout=0.1)

    # Create data with mismatched input dims
    data = Data(
        x=torch.rand(5, 8),      # wrong input_dim (8 instead of 10)
        topo=torch.rand(5, 6),
        edge_index=torch.tensor([[0,1],[1,0]], dtype=torch.long),
        batch=torch.zeros(5, dtype=torch.long)
    )

    with pytest.raises(RuntimeError):
        model(data)
