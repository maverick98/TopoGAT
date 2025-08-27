import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger()

class TopoEmbeddingVisualizer:
    """
    Visualizes graph node embeddings using PCA or t-SNE, supporting multiple embedding types.
    """

    def __init__(self, method='tsne', save_dir='plots'):
        assert method in ['tsne', 'pca'], "Method must be 'tsne' or 'pca'"
        self.method = method
        self.save_dir = os.path.join(save_dir, method)
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set(style='whitegrid')

    def _reduce_dim(self, X):
        X_scaled = StandardScaler().fit_transform(X)
        if self.method == 'tsne':
            return TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X_scaled)
        return PCA(n_components=2).fit_transform(X_scaled)

    def _plot_and_save(self, X_reduced, y, title, filename):
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", np.unique(y).max() + 1)
        sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette=palette, legend='full', s=30)
        plt.title(title, fontsize=14)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved: {save_path}")

    def visualize(self, data, model=None, mode=None):
        """
        Visualizes embeddings using PCA/t-SNE. Requires model if mode is 'latent1' or 'latent2'.

        Args:
            data: PyG data object with .x, .y, .topo, etc.
            model: Model with forward hooks populating 'x_gat1', 'x_gat2'.
            mode: 'x', 'topo', 'combined', 'latent1', 'latent2' or None (to visualize all).
        """
        if model and (mode in ['latent1', 'latent2'] or mode is None):
            model.eval()
            with torch.no_grad():
                model(data)

        # Safe getters
        get_tensor = lambda attr: getattr(data, attr, torch.empty((data.num_nodes, 0)))

        modes = {
            'x': data.x.cpu().numpy(),
            'topo': get_tensor('topo').cpu().numpy(),
            'combined': torch.cat([data.x, get_tensor('topo')], dim=1).cpu().numpy(),
            'latent1': get_tensor('x_gat1').cpu().detach().numpy(),
            'latent2': get_tensor('x_gat2').cpu().detach().numpy(),
        }

        y = data.y.cpu().numpy()

        # Single-mode or all-mode
        if mode:
            if mode not in modes:
                raise ValueError(f"Invalid mode '{mode}'. Choose from {list(modes.keys())}")
            modes = {mode: modes[mode]}

        for m, X in modes.items():
            try:
                if X.shape[1] == 0 or np.isnan(X).any() or np.isinf(X).any() or np.std(X) == 0:
                    logger.warning(f"Skipping '{m}': invalid or constant data")
                    continue
                X_reduced = self._reduce_dim(X)
                self._plot_and_save(X_reduced, y, f"TopoGNN Embeddings ({m})", f"{m}_{self.method}.png")
            except Exception as e:
                logger.error(f"Error visualizing '{m}': {e}")
