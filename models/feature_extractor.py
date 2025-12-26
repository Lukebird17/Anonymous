"""
特征提取器
提取传统的图拓扑特征用于基准实验
"""

import networkx as nx
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """传统图特征提取器"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_node_features(self, G: nx.Graph, nodes: List = None) -> np.ndarray:
        """
        提取节点的拓扑特征
        
        Args:
            G: NetworkX图
            nodes: 节点列表，如果为None则提取所有节点
            
        Returns:
            特征矩阵 [n_nodes, n_features]
        """
        if nodes is None:
            nodes = list(G.nodes())
        
        logger.info(f"提取 {len(nodes)} 个节点的拓扑特征...")
        
        features = []
        self.feature_names = [
            'degree',
            'in_degree',
            'out_degree',
            'degree_centrality',
            'betweenness_centrality',
            'closeness_centrality',
            'pagerank',
            'clustering_coefficient',
            'triangles',
            'square_clustering'
        ]
        
        # 预计算所有节点的特征
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        closeness_centrality = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # 聚集系数
        if G.is_directed():
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
            triangles_dict = nx.triangles(G_undirected)
            square_clustering_dict = nx.square_clustering(G_undirected)
        else:
            clustering = nx.clustering(G)
            triangles_dict = nx.triangles(G)
            square_clustering_dict = nx.square_clustering(G)
        
        # 对每个节点提取特征
        for node in nodes:
            node_features = [
                G.degree(node),
                G.in_degree(node) if G.is_directed() else G.degree(node),
                G.out_degree(node) if G.is_directed() else G.degree(node),
                degree_centrality.get(node, 0),
                betweenness_centrality.get(node, 0),
                closeness_centrality.get(node, 0),
                pagerank.get(node, 0),
                clustering.get(node, 0),
                triangles_dict.get(node, 0),
                square_clustering_dict.get(node, 0)
            ]
            features.append(node_features)
        
        features = np.array(features)
        logger.info(f"特征矩阵形状: {features.shape}")
        
        return features
    
    def extract_neighborhood_features(self, G: nx.Graph, node, k: int = 2) -> Dict:
        """
        提取节点的k-hop邻居特征
        
        Args:
            G: NetworkX图
            node: 目标节点
            k: 邻居跳数
            
        Returns:
            邻居特征字典
        """
        features = {}
        
        # 1-hop邻居
        neighbors_1 = set(G.neighbors(node))
        features['n_neighbors_1'] = len(neighbors_1)
        
        # 2-hop邻居
        if k >= 2:
            neighbors_2 = set()
            for n in neighbors_1:
                neighbors_2.update(G.neighbors(n))
            neighbors_2 -= neighbors_1
            neighbors_2.discard(node)
            features['n_neighbors_2'] = len(neighbors_2)
        
        # 邻居的度分布
        neighbor_degrees = [G.degree(n) for n in neighbors_1]
        if neighbor_degrees:
            features['neighbor_degree_mean'] = np.mean(neighbor_degrees)
            features['neighbor_degree_std'] = np.std(neighbor_degrees)
            features['neighbor_degree_max'] = np.max(neighbor_degrees)
            features['neighbor_degree_min'] = np.min(neighbor_degrees)
        else:
            features['neighbor_degree_mean'] = 0
            features['neighbor_degree_std'] = 0
            features['neighbor_degree_max'] = 0
            features['neighbor_degree_min'] = 0
        
        return features
    
    def extract_motif_features(self, G: nx.Graph, node) -> Dict:
        """
        提取节点的Motif特征（局部结构模式）
        
        Args:
            G: NetworkX图
            node: 目标节点
            
        Returns:
            Motif特征字典
        """
        features = {}
        
        # 三角形数量
        if G.is_directed():
            G_undirected = G.to_undirected()
            features['triangles'] = nx.triangles(G_undirected, node)
        else:
            features['triangles'] = nx.triangles(G, node)
        
        # 聚集系数
        features['clustering'] = nx.clustering(G, node)
        
        # 邻居之间的边数
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subgraph = G.subgraph(neighbors)
            features['neighbor_edges'] = subgraph.number_of_edges()
        else:
            features['neighbor_edges'] = 0
        
        return features
    
    def compute_structural_similarity(self, G: nx.Graph, 
                                     node1, node2) -> float:
        """
        计算两个节点的结构相似度
        
        Args:
            G: NetworkX图
            node1: 节点1
            node2: 节点2
            
        Returns:
            相似度分数 [0, 1]
        """
        # Jaccard相似度（基于共同邻居）
        neighbors1 = set(G.neighbors(node1))
        neighbors2 = set(G.neighbors(node2))
        
        if len(neighbors1) == 0 and len(neighbors2) == 0:
            return 1.0
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_degree_sequence_signature(self, G: nx.Graph, node, 
                                     k: int = 2) -> List[int]:
        """
        获取节点的度序列签名（用于结构匹配）
        
        Args:
            G: NetworkX图
            node: 节点
            k: 邻居跳数
            
        Returns:
            度序列签名
        """
        signature = []
        
        # 自身的度
        signature.append(G.degree(node))
        
        # 邻居的度序列（排序后）
        neighbors = list(G.neighbors(node))
        neighbor_degrees = sorted([G.degree(n) for n in neighbors], reverse=True)
        signature.extend(neighbor_degrees[:10])  # 只取前10个
        
        # 2-hop邻居的度统计
        if k >= 2:
            neighbors_2 = set()
            for n in neighbors:
                neighbors_2.update(G.neighbors(n))
            neighbors_2 -= set(neighbors)
            neighbors_2.discard(node)
            
            if neighbors_2:
                degrees_2 = [G.degree(n) for n in neighbors_2]
                signature.extend([
                    len(neighbors_2),
                    int(np.mean(degrees_2)),
                    int(np.max(degrees_2))
                ])
        
        return signature


def main():
    """主函数示例"""
    import pickle
    from pathlib import Path
    
    # 加载图
    graph_path = Path("data/processed/github_graph.gpickle")
    if not graph_path.exists():
        logger.error(f"找不到文件: {graph_path}")
        return
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 提取特征
    extractor = FeatureExtractor()
    
    nodes = list(G.nodes())[:1000]  # 只取前1000个节点作为示例
    features = extractor.extract_node_features(G, nodes)
    
    # 保存特征
    output_path = Path("models") / "traditional_features.npy"
    output_path.parent.mkdir(exist_ok=True)
    np.save(output_path, features)
    logger.info(f"特征已保存到: {output_path}")
    
    # 打印特征名称
    print("\n特征列表:")
    for i, name in enumerate(extractor.feature_names):
        print(f"{i+1}. {name}")


if __name__ == "__main__":
    main()


