import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.transforms import NormalizeFeatures
from utils.topology import TopologyProcessor
'''
    Imports required libraries.

    Planetoid and TUDataset are standard graph datasets (e.g., Cora).

    NormalizeFeatures normalizes node features to unit norm.

    TopologyProcessor is  custom class that computes persistent homology features.

'''

class TopoGraphDataset:
    """
        TopoGraphDataset class handles loading of standard graph datasets (e.g., Cora, PubMed, Citeseer, or TUDatasets),
        applies normalization, and augments node features with topological descriptors computed via persistent homology.

        Attributes:
            name (str): Name of the dataset to load.
            dataset: The loaded graph dataset.
            processor (TopologyProcessor): Utility to compute persistence-based topological features.

        Methods:
            get_data(): Returns the data object with topological features added, 
                        the total number of input features (original + topological), 
                        and the number of target classes.
    """

    def __init__(self, name="Cora"):
        """
        Initializes the dataset. If it's a Planetoid dataset (Cora, PubMed, Citeseer),
        it loads from PyG's built-in Planetoid datasets. Otherwise, it uses TUDataset.

        Args:
            name (str): Name of the dataset to load. Default is 'Cora'.
        """

        '''
            If the dataset is one of the Planetoid ones, it uses the Planetoid loader.
            Applies feature normalization as a preprocessing transform.
            Downloads the dataset into the data/{name} directory if not already present.

            If it's not Planetoid, it assumes a TU Dortmund dataset and loads it similarly.
        '''  
        if name in ["Cora", "PubMed", "Citeseer"]:
            self.dataset = Planetoid(root=f'data/{name}', name=name, transform=NormalizeFeatures())
          
        else:
            self.dataset = TUDataset(root=f'data/{name}', name=name)
            
        self.processor = TopologyProcessor()
        #Creates an instance of your TopologyProcessor to compute persistent homology features.
        

    def get_data(self):
         """
        Loads the graph data and augments node features with persistent homology.

        Returns:
            data (torch_geometric.data.Data): The graph object with topo features.
            in_channels (int): Total input feature size after topo concatenation.
            num_classes (int): Number of target labels in the dataset.
        """
        data = self.dataset[0]
        # Compute persistent homology features for each node
        data.topo = self.processor.compute_persistence_diagrams(data)
        # Return the graph, feature size, and number of classes
        return data, data.num_node_features + data.topo.shape[1], self.dataset.num_classes