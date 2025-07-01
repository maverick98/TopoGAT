import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import MLP

class BaseGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()

        # GINConv layers use MLPs internally
        self.mlp1 = MLP([input_dim, hidden_dim, hidden_dim], norm=None)
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = MLP([hidden_dim, hidden_dim, hidden_dim], norm=None)
        self.conv2 = GINConv(self.mlp2)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        graph_embed = global_mean_pool(x, batch)
        return F.log_softmax(self.classifier(graph_embed), dim=-1)
