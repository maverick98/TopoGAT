# models/topogin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, MLP
from abc import ABC, abstractmethod
from typing import Any
from models.registry import register_topogin


class GatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GatedMLP, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.linear(x))
        g = torch.sigmoid(self.gate(x))
        x = h * g
        return self.output_layer(x)


class BaseTopoGIN(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data):
        pass


@register_topogin("topogin-gatedmlp")
class TopoGatedMLPGIN(BaseTopoGIN):
    def __init__(self, input_dim: int, topo_dim: int, hidden_dim: int = 64, heads: int = 1, num_classes: int = 2):
        super().__init__()

        self.topo_mlp = GatedMLP(topo_dim, hidden_dim, input_dim)

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Any):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Sanitize topo_feat before using
        if hasattr(data, 'topo_feat') and data.topo_feat is not None:
            topo_feat = data.topo_feat

            # Replace NaN and Inf with 0
            topo_feat = torch.nan_to_num(topo_feat, nan=0.0, posinf=0.0, neginf=0.0)

            topo = self.topo_mlp(topo_feat)
            x = torch.cat([x, topo], dim=1)
        else:
            topo = torch.zeros_like(x)
            x = torch.cat([x, topo], dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Clamp final logits to prevent exploding values
        x = torch.clamp(x, min=-10, max=10)

        return x
