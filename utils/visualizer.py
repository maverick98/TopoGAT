import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class TopoEmbeddingVisualizer:
    """
    Visualizes embeddings (raw features, PH embeddings, combined, GAT layer outputs)
    using PCA or t-SNE and saves publication-quality plots.
    """

    def __init__(self, method='tsne', save_dir='plots'):
        """
        Initializes the visualizer with a dimensionality reduction method and save directory.

        Args:
            method (str): 'tsne' or 'pca' for dimensionality reduction.
            save_dir (str): Directory where plots will be saved.
        """
        assert method in ['tsne', 'pca'], "Method must be 'tsne' or 'pca'"
        self.method = method
        self.save_dir = os.path.join(save_dir, method)
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set(style='whitegrid')

    def _reduce_dim(self, X):
        """
        Standardizes and reduces dimensionality of input features using PCA or t-SNE.

        Args:
            X (ndarray): Input feature matrix [num_nodes, num_features].

        Returns:
            ndarray: 2D reduced feature matrix [num_nodes, 2].
        """
        X_scaled = StandardScaler().fit_transform(X)
        if self.method == 'tsne':
            return TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X_scaled)
        else:
            return PCA(n_components=2).fit_transform(X_scaled)

    def _plot_and_save(self, X_reduced, y, title, filename):
        """
        Plots the reduced embeddings and saves them to a file.

        Args:
            X_reduced (ndarray): 2D embeddings [num_nodes, 2].
            y (ndarray): Class labels [num_nodes].
            title (str): Title of the plot.
            filename (str): Filename to save the plot.
        """
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", np.unique(y).max() + 1)
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette=palette, legend='full', s=30)
        plt.title(title, fontsize=14)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")

    def visualize(self, data, model=None, mode=None):
        """
        Visualizes selected mode (or all if None). Valid modes: x, topo, combined, latent1, latent2
        Requires `model` only for latent modes.

        Args:
            data: PyG data object with x, y, topo, etc.
            model: GNN model with forward hooks for 'x_gat1', 'x_gat2' if required.
            mode (str or None): Which embedding to visualize.
        """
        if model and (mode in ['latent1', 'latent2'] or mode is None):
            model.eval()
            with torch.no_grad():
                model(data)

        modes = {
            'x': data.x.cpu().numpy(),
            'topo': getattr(data, 'topo', torch.empty((data.num_nodes, 0))).cpu().numpy(),
            'combined': torch.cat([data.x, getattr(data, 'topo', torch.empty((data.num_nodes, 0)))], dim=1).cpu().numpy(),
            'latent1': getattr(data, 'x_gat1', torch.empty((data.num_nodes, 0))).cpu().detach().numpy(),
            'latent2': getattr(data, 'x_gat2', torch.empty((data.num_nodes, 0))).cpu().detach().numpy()
        }

        y = data.y.cpu().numpy()

        if mode:
            if mode not in modes:
                raise ValueError(f"Invalid mode '{mode}'. Choose from {list(modes.keys())}")
            modes = {mode: modes[mode]}

        for m, X in modes.items():
            try:
                if X.shape[1] == 0 or np.isnan(X).any() or np.isinf(X).any() or np.std(X) == 0:
                    logger.warning(f"Skipping {m}: invalid or constant data")
                    continue
                X_reduced = self._reduce_dim(X)
                self._plot_and_save(X_reduced, y, f"TopoGAT Embeddings ({m})", f"{m}_{self.method}.png")
            except Exception as e:
                logger.error(f"Error visualizing {m}: {e}")
