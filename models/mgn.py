"""
MGN model和同质图训练工具，基于提供的参考实现（MGN_model.py, Graph_Dataset.py）。
"""

import numpy as np
import torch
from torch import nn
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_layers, layer_normalized=True):
        super().__init__()
        modules = []
        for l in range(hidden_layers):
            modules.append(Linear(input_dim if l == 0 else latent_dim, latent_dim))
            modules.append(ReLU())
        modules.append(Linear(latent_dim if hidden_layers > 0 else input_dim, output_dim))
        if layer_normalized:
            modules.append(LayerNorm(output_dim, elementwise_affine=False))
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class GraphNetBlock(MessagePassing):
    def __init__(self, latent_dim, hidden_layers, skip_connection=True):
        super().__init__(aggr='sum')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layers, False)
        self.mlp_edge_info = MLP(3 * latent_dim, latent_dim, latent_dim, hidden_layers, False)
        self.skip_connection = skip_connection

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        aggr_edge_attr = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        out = self.mlp_node_delta(torch.cat([x_dst, aggr_edge_attr], dim=-1))
        if self.skip_connection:
            out = out + x_dst
        return out

    def message(self, x_i, x_j, edge_attr):
        return self.mlp_edge_info(torch.cat([edge_attr, x_i, x_j], dim=-1))


class MGNModel(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, latent_dim=128, mgn_layers=2, mlp_hidden_layers=1):
        super().__init__()
        self.node_encoder = MLP(node_dim, latent_dim, latent_dim, mlp_hidden_layers, False)
        self.edge_encoder = MLP(edge_dim, latent_dim, latent_dim, mlp_hidden_layers, False)
        self.layers = nn.ModuleList(
            [GraphNetBlock(latent_dim, mlp_hidden_layers) for _ in range(mgn_layers)]
        )
        self.decoder = MLP(latent_dim, latent_dim, out_dim, mlp_hidden_layers, False)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.decoder(x)


def _pad_features_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] >= target_dim:
        return vec[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: vec.shape[0]] = vec
    return padded


def build_homogeneous_data(
    G,
    node_features: Dict,
    node_labels: Dict,
    input_dim: int,
    edge_attr_dim: int = 1,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[Data, List[int], List[int]]:
    """Convert a NetworkX graph to PyG Data for MGN."""
    rng = np.random.default_rng(seed)
    nodes = sorted([n for n in G.nodes() if n in node_features])
    node_id_map = {n: i for i, n in enumerate(nodes)}
    x = np.zeros((len(nodes), input_dim), dtype=np.float32)
    y = np.full((len(nodes),), -1, dtype=np.int64)

    for n in nodes:
        feat = node_features[n]
        feat = _pad_features_to_dim(np.asarray(feat, dtype=np.float32), input_dim)
        x[node_id_map[n]] = feat
        if n in node_labels:
            y[node_id_map[n]] = int(node_labels[n])

    edges = []
    for u, v in G.edges():
        if u in node_id_map and v in node_id_map:
            edges.append((node_id_map[u], node_id_map[v]))
            edges.append((node_id_map[v], node_id_map[u]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.ones((edge_index.size(1), edge_attr_dim), dtype=torch.float)

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.long),
    )

    labeled_indices = [node_id_map[n] for n in nodes if n in node_labels]
    rng.shuffle(labeled_indices)
    n_train = max(1, int(len(labeled_indices) * train_ratio)) if labeled_indices else 0
    train_idx = labeled_indices[:n_train]
    test_idx = labeled_indices[n_train:] if labeled_indices else []

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    if train_idx:
        data.train_mask[train_idx] = True
    if test_idx:
        data.test_mask[test_idx] = True

    return data, train_idx, test_idx


class MGNTrainer:
    """Trainer for homogeneous MGN."""

    def __init__(self, data: Data, num_classes: int, edge_attr_dim: int = 1,
                 latent_dim: int = 128, mgn_layers: int = 2, mlp_hidden_layers: int = 1,
                 learning_rate: float = 5e-4, device: str = 'cpu'):
        self.data = data
        self.device = torch.device(device)
        self.model = MGNModel(
            node_dim=data.x.size(1),
            edge_dim=edge_attr_dim,
            out_dim=num_classes,
            latent_dim=latent_dim,
            mgn_layers=mgn_layers,
            mlp_hidden_layers=mlp_hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs: int = 50):
        self.data = self.data.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            mask = self.data.train_mask & (self.data.y >= 0)
            if mask.sum() == 0:
                return
            loss = self.criterion(out[mask], self.data.y[mask])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"MGN epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    def evaluate(self, mask_name: str = 'test_mask') -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
            mask = getattr(self.data, mask_name) & (self.data.y >= 0)
            if mask.sum() == 0:
                return 0.0, np.array([]), np.array([])
            pred = out.argmax(dim=1)
            y_true = self.data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            acc = (pred[mask] == self.data.y[mask]).float().mean().item()
        return acc, y_true, y_pred
