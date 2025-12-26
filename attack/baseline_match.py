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
    
    def __init__(self, similarity_metric: str = 'cosine'):
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
    
    def match_greedy(self, similarity_matrix: np.ndarray) -> Dict[int, int]:
        """贪心匹配"""
        predictions = {}
        for i in range(similarity_matrix.shape[0]):
            best_match = np.argmax(similarity_matrix[i])
            predictions[i] = best_match
        return predictions
    
    def match_hungarian(self, similarity_matrix: np.ndarray) -> Dict[int, int]:
        """匈牙利算法匹配"""
        cost_matrix = -similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        predictions = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        return predictions


