"""
图对齐算法
将匿名图的嵌入空间映射到原始图的嵌入空间
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAligner:
    """图对齐器"""
    
    def __init__(self):
        self.alignment_matrix = None
    
    def align_procrustes(self, embeddings1: np.ndarray,
                        embeddings2: np.ndarray,
                        seed_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        使用Procrustes方法进行图对齐
        
        Args:
            embeddings1: 匿名图嵌入 [n_anon, dim]
            embeddings2: 原始图嵌入 [n_orig, dim]
            seed_pairs: 种子节点对 [(匿名索引, 原始索引), ...]
            
        Returns:
            对齐后的匿名图嵌入
        """
        logger.info(f"使用 {len(seed_pairs)} 个种子节点进行Procrustes对齐...")
        
        # 提取种子节点的嵌入
        anon_seeds = np.array([embeddings1[anon_idx] for anon_idx, _ in seed_pairs])
        orig_seeds = np.array([embeddings2[orig_idx] for _, orig_idx in seed_pairs])
        
        # 计算Procrustes变换矩阵
        R, _ = orthogonal_procrustes(anon_seeds, orig_seeds)
        self.alignment_matrix = R
        
        # 对齐所有匿名节点的嵌入
        aligned_embeddings = embeddings1 @ R
        
        logger.info("对齐完成")
        return aligned_embeddings
    
    def align_linear(self, embeddings1: np.ndarray,
                    embeddings2: np.ndarray,
                    seed_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        使用线性变换进行图对齐
        
        Args:
            embeddings1: 匿名图嵌入
            embeddings2: 原始图嵌入
            seed_pairs: 种子节点对
            
        Returns:
            对齐后的嵌入
        """
        logger.info(f"使用 {len(seed_pairs)} 个种子节点进行线性对齐...")
        
        anon_seeds = np.array([embeddings1[anon_idx] for anon_idx, _ in seed_pairs])
        orig_seeds = np.array([embeddings2[orig_idx] for _, orig_idx in seed_pairs])
        
        # 使用最小二乘法求解变换矩阵 W: X @ W = Y
        W = np.linalg.lstsq(anon_seeds, orig_seeds, rcond=None)[0]
        self.alignment_matrix = W
        
        # 应用变换
        aligned_embeddings = embeddings1 @ W
        
        logger.info("对齐完成")
        return aligned_embeddings


