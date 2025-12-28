import torch
import networkx as nx
import numpy as np
from config import *

def compute_features(G, feature_keys):

    if 'degree_centrality' in feature_keys:
        dc = nx.degree_centrality(G)
        nx.set_node_attributes(G, dc, 'degree_centrality')

    if 'betweenness_centrality' in feature_keys:
        bc = nx.betweenness_centrality(G)
        nx.set_node_attributes(G, bc, 'betweenness_centrality')

        # 接近中心性
    if 'closeness_centrality' in feature_keys:
        cc = nx.closeness_centrality(G)
        nx.set_node_attributes(G, cc, 'closeness_centrality')

        # PageRank
    if 'pagerank' in feature_keys:
        pr = nx.pagerank(G)
        nx.set_node_attributes(G, pr, 'pagerank')

        # 聚类系数 (NetworkX 会自动处理 DiGraph)
    if 'clustering' in feature_keys:
        clus = nx.clustering(G)
        nx.set_node_attributes(G, clus, 'clustering')

    if 'in_degree' in feature_keys:
        in_deg = dict(G.in_degree())
        nx.set_node_attributes(G, in_deg, 'in_degree')

    if 'out_degree' in feature_keys:
        out_deg = dict(G.out_degree())
        nx.set_node_attributes(G, out_deg, 'out_degree')

    node_list = list(G.nodes())
    node_mapping = {node: i for i, node in enumerate(node_list)}

    features_list = []
    for node in node_list:
        f = []
        for k in feature_keys:
            val = G.nodes[node].get(k, 0.0)
            f.append(val)
        features_list.append(f)

    x = torch.tensor(features_list, dtype=torch.float)

    if x.size(0) > 1:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - mean) / (std + 1e-6)

    edges = list(G.edges())
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in edges], dtype=torch.long)

    if len(edges) > 0:
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return x, edge_index, node_mapping