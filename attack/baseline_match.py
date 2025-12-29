"""
基于传统特征的基准匹配算法
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.optimize import linear_sum_assignment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineMatcher:
    """基于传统特征的基准匹配器"""
    
    def __init__(self, G_orig: nx.Graph = None, G_anon: nx.Graph = None, 
                 similarity_metric: str = 'cosine'):
        """
        初始化基准匹配器
        
        Args:
            G_orig: 原始图（可选）
            G_anon: 匿名图（可选）
            similarity_metric: 相似度度量方式
        """
        self.G_orig = G_orig
        self.G_anon = G_anon
        self.similarity_metric = similarity_metric
    
    def extract_features(self, G: nx.Graph, nodes: List) -> np.ndarray:
        """提取节点特征"""
        from models.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        features = extractor.extract_node_features(G, nodes)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        return features
    
    def compute_similarity_matrix(self, features1: np.ndarray, 
                                  features2: np.ndarray) -> np.ndarray:
        """计算相似度矩阵"""
        if self.similarity_metric == 'cosine':
            similarity = cosine_similarity(features1, features2)
        else:
            distances = euclidean_distances(features1, features2)
            similarity = 1 / (1 + distances)
        return similarity
    
    def match_greedy(self, similarity_matrix: np.ndarray, top_k: int = 10) -> Dict[int, List[int]]:
        """
        贪心匹配，返回top-k候选
        
        Args:
            similarity_matrix: 相似度矩阵 [n_orig, n_anon]
            top_k: 返回前k个候选
            
        Returns:
            {orig_idx: [top_k_anon_indices]}
        """
        predictions = {}
        for i in range(similarity_matrix.shape[0]):
            # 获取top-k个最相似的候选
            top_indices = np.argsort(similarity_matrix[i])[::-1][:top_k]
            predictions[i] = top_indices.tolist()
        return predictions
    
    def match_hungarian(self, similarity_matrix: np.ndarray) -> Dict[int, int]:
        """匈牙利算法匹配"""
        cost_matrix = -similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        predictions = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        return predictions
    
    def match_by_features(self, G_orig: nx.Graph = None, G_anon: nx.Graph = None, 
                         top_k: int = 10) -> Dict[int, List[int]]:
        """
        使用特征匹配进行去匿名化
        
        Args:
            G_orig: 原始图（如果初始化时没提供）
            G_anon: 匿名图（如果初始化时没提供）
            top_k: 返回前k个候选
            
        Returns:
            匹配结果字典 {原始节点: [top-k匿名节点]}
        """
        if G_orig is None:
            G_orig = self.G_orig
        if G_anon is None:
            G_anon = self.G_anon
        
        if G_orig is None or G_anon is None:
            raise ValueError("需要提供原始图和匿名图")
        
        # 提取特征
        nodes_orig = sorted(list(G_orig.nodes()))
        nodes_anon = sorted(list(G_anon.nodes()))
        
        features_orig = self.extract_features(G_orig, nodes_orig)
        features_anon = self.extract_features(G_anon, nodes_anon)
        
        # 计算相似度
        similarity = self.compute_similarity_matrix(features_orig, features_anon)
        
        # 匹配
        matches = self.match_greedy(similarity, top_k)
        
        # 转换为原始节点ID
        predictions = {}
        for i in range(len(nodes_orig)):
            orig_node = nodes_orig[i]
            anon_nodes = [nodes_anon[idx] for idx in matches[i] if idx < len(nodes_anon)]
            predictions[orig_node] = anon_nodes
        
        return predictions


