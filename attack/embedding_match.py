"""
基于嵌入的匹配算法
使用DeepWalk等图嵌入方法进行节点匹配
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingMatcher:
    """基于嵌入的匹配器"""
    
    def __init__(self):
        pass
    
    def compute_similarity_matrix(self, embeddings1: np.ndarray,
                                  embeddings2: np.ndarray) -> np.ndarray:
        """
        计算嵌入相似度矩阵
        
        Args:
            embeddings1: 匿名图节点嵌入 [n_anon, dim]
            embeddings2: 原始图节点嵌入 [n_orig, dim]
            
        Returns:
            相似度矩阵 [n_anon, n_orig]
        """
        similarity = cosine_similarity(embeddings1, embeddings2)
        return similarity
    
    def match_greedy(self, similarity_matrix: np.ndarray) -> Dict[int, int]:
        """贪心匹配"""
        predictions = {}
        for i in range(similarity_matrix.shape[0]):
            best_match = np.argmax(similarity_matrix[i])
            predictions[i] = best_match
        return predictions
    
    def match_with_seeds(self, similarity_matrix: np.ndarray,
                        seed_pairs: List[Tuple[int, int]]) -> Dict[int, int]:
        """
        基于种子节点的匹配
        
        Args:
            similarity_matrix: 相似度矩阵
            seed_pairs: 种子节点对 [(匿名索引, 原始索引), ...]
            
        Returns:
            匹配结果
        """
        predictions = {}
        
        # 添加种子节点
        for anon_idx, orig_idx in seed_pairs:
            predictions[anon_idx] = orig_idx
        
        # 调整相似度矩阵
        adjusted_similarity = similarity_matrix.copy()
        
        # 已匹配的节点设为极低相似度
        for anon_idx, orig_idx in seed_pairs:
            adjusted_similarity[anon_idx, :] = -np.inf
            adjusted_similarity[:, orig_idx] = -np.inf
            adjusted_similarity[anon_idx, orig_idx] = np.inf
        
        # 对剩余节点进行匹配
        for i in range(similarity_matrix.shape[0]):
            if i not in predictions:
                best_match = np.argmax(adjusted_similarity[i])
                predictions[i] = best_match
        
        return predictions


