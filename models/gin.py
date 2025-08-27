#models/gin.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import MLP

class BaseGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.2, use_log_softmax=True):
        super().__init__()
        print(f"[DEBUG] BaseGIN: input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}")
        print(f"[DEBUG] Type of input_dim: {type(input_dim)}")

        self.use_log_softmax = use_log_softmax
        self.dropout = dropout

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
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embed = global_mean_pool(x, batch)
        logits = self.classifier(graph_embed)

        if self.use_log_softmax:
            return F.log_softmax(logits, dim=-1)
        else:
            return logits
