import torch
from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data)
            pred = logits.argmax(dim=1)
            acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), pred[self.data.test_mask].cpu())
        return acc