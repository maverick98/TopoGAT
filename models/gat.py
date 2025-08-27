#models/gat.py
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GATConv

import torch.nn as nn
import os
from datetime import datetime
from torch_geometric.data import Batch
from torch_geometric.utils import degree
import numpy as np
import random
import matplotlib.pyplot as plt

class BaseGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, heads=1):
        super().__init__()
        print(f"[DEBUG] BaseGAT: input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}, heads={heads}")
        print(f"[DEBUG] Type of input_dim: {type(input_dim)}")

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        graph_embed = global_mean_pool(x, data.batch)
        return F.log_softmax(self.classifier(graph_embed), dim=-1)

