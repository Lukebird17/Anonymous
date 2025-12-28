import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr='mean')
        self.dropout = dropout

    def forward_single(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, dim=1, p=2)
        return x

    def forward(self, data_orig, data_anon):
        emb_orig = self.forward_single(data_orig.x, data_orig.edge_index)
        emb_anon = self.forward_single(data_anon.x, data_anon.edge_index)
        return emb_orig, emb_anon
