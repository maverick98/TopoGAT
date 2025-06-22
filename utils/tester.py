import unittest
import torch
from models.topogat import TopoGAT

class TestTopoGAT(unittest.TestCase):
    def test_forward_shape(self):
        model = TopoGAT(1433 + 8, 7)
        x = torch.rand((2708, 1433 + 8))
        edge_index = torch.randint(0, 2708, (2, 10556))
        out = model.forward(type('Data', (object,), {'x': x, 'edge_index': edge_index, 'topo': torch.rand(2708, 8)}))
        self.assertEqual(out.shape, (2708, 7))

if __name__ == '__main__':
    unittest.main()