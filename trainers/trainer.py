import torch
import torch.nn.functional as F
import yaml
from utils.logger import get_logger
from data.dataset import TopoGraphDataset
from models.topogat import TopoGAT

logger = get_logger()

class Trainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.epochs = config['epochs']
        self.data, in_channels, out_channels = TopoGraphDataset(config['dataset']).get_data()
        self.model = TopoGAT(in_channels, out_channels, config['topological_dim'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            if epoch % 20 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        torch.save(self.model.state_dict(), 'model.pt')