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
    
    def __init__(self, G_orig=None, G_anon=None):
        """
        初始化嵌入匹配器
        
        Args:
            G_orig: 原始图（可选，为了兼容性）
            G_anon: 匿名图（可选，为了兼容性）
        """
        self.G_orig = G_orig
        self.G_anon = G_anon
        self.embeddings_orig = None
        self.embeddings_anon = None
    
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
    
    def match_by_similarity(self, top_k: int = 10) -> Dict[int, List[int]]:
        """
        基于余弦相似度进行匹配（使用已存储的嵌入）
        
        Args:
            top_k: 考虑前k个候选
            
        Returns:
            匹配结果 {原始节点索引: [top-k匿名节点索引]}
        """
        if self.embeddings_orig is None or self.embeddings_anon is None:
            raise ValueError("嵌入未设置，请先训练DeepWalk并设置embeddings_orig和embeddings_anon")
        
        # 计算相似度矩阵 [n_orig, n_anon]
        # 注意：第一个参数是行，第二个参数是列
        similarity_matrix = self.compute_similarity_matrix(
            self.embeddings_orig, self.embeddings_anon  # 修复：orig在前，anon在后
        )
        
        # 使用贪心匹配
        predictions = self.match_greedy(similarity_matrix, top_k)
        
        logger.info(f"基于嵌入的匹配完成，共 {len(predictions)} 对")
        return predictions
    
    def match_with_seeds(self, seed_mapping: Dict[int, int], top_k: int = 10) -> Dict[int, List[int]]:
        """
        基于种子节点的对齐匹配（使用已存储的嵌入）
        
        Args:
            seed_mapping: 种子节点映射 {原始节点索引: 匿名节点索引}
            top_k: 考虑前k个候选
            
        Returns:
            匹配结果 {原始节点索引: [top-k匿名节点索引]}
        """
        if self.embeddings_orig is None or self.embeddings_anon is None:
            raise ValueError("嵌入未设置")
        
        # 计算相似度矩阵 [n_orig, n_anon]
        similarity_matrix = self.compute_similarity_matrix(
            self.embeddings_orig, self.embeddings_anon  # 修复：orig在前，anon在后
        )
        
        # 使用种子对齐
        predictions = {}
        
        # 调整相似度矩阵
        adjusted_similarity = similarity_matrix.copy()
        
        # 已匹配的节点设为极低相似度
        for orig_idx, anon_idx in seed_mapping.items():
            adjusted_similarity[orig_idx, :] = -np.inf
            adjusted_similarity[:, anon_idx] = -np.inf
            adjusted_similarity[orig_idx, anon_idx] = np.inf
            # 种子节点直接添加为top-1
            predictions[orig_idx] = [anon_idx]
        
        # 对剩余节点进行匹配
        for orig_idx in range(similarity_matrix.shape[0]):
            if orig_idx not in predictions:
                # 获取top-k个最相似的候选
                top_indices = np.argsort(adjusted_similarity[orig_idx])[::-1][:top_k]
                predictions[orig_idx] = top_indices.tolist()
        
        logger.info(f"种子对齐匹配完成，使用 {len(seed_mapping)} 个种子，共 {len(predictions)} 对")
        return predictions


