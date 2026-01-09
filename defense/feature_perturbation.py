"""
特征扰动防御 - 对节点特征添加噪声
实现多种特征扰动策略以保护隐私
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional
import random


class FeaturePerturbationDefense:
    """特征扰动防御器"""
    
    def __init__(self, G: nx.Graph, noise_level: float = 0.1):
        """
        初始化特征扰动防御器
        
        Args:
            G: 原始图（包含节点特征）
            noise_level: 噪声水平 (0-1)
        """
        self.G = G.copy()
        self.noise_level = noise_level
        self.n_nodes = G.number_of_nodes()
    
    def apply_gaussian_noise(self, seed: Optional[int] = None) -> nx.Graph:
        """
        应用高斯噪声到节点特征
        
        Args:
            seed: 随机种子
            
        Returns:
            特征扰动后的图
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n应用高斯噪声 (噪声水平: {self.noise_level})")
        
        G_perturbed = self.G.copy()
        nodes_with_features = 0
        
        for node in G_perturbed.nodes():
            if 'feature' in G_perturbed.nodes[node]:
                # 获取原始特征
                original_feature = np.array(G_perturbed.nodes[node]['feature'])
                
                # 添加高斯噪声
                noise = np.random.normal(0, self.noise_level, size=original_feature.shape)
                perturbed_feature = original_feature + noise
                
                # 归一化（如果原始特征是归一化的）
                if np.all(original_feature >= 0) and np.all(original_feature <= 1):
                    perturbed_feature = np.clip(perturbed_feature, 0, 1)
                
                G_perturbed.nodes[node]['feature'] = perturbed_feature
                nodes_with_features += 1
        
        print(f"  - 扰动节点数: {nodes_with_features}")
        
        return G_perturbed
    
    def apply_feature_masking(self, mask_ratio: float = 0.3, seed: Optional[int] = None) -> nx.Graph:
        """
        随机遮罩部分特征维度
        
        Args:
            mask_ratio: 遮罩比例
            seed: 随机种子
            
        Returns:
            特征遮罩后的图
        """
        if seed is not None:
            random.seed(seed)
        
        print(f"\n应用特征遮罩 (遮罩比例: {mask_ratio})")
        
        G_masked = self.G.copy()
        nodes_with_features = 0
        total_masked = 0
        
        for node in G_masked.nodes():
            if 'feature' in G_masked.nodes[node]:
                # 获取原始特征
                original_feature = np.array(G_masked.nodes[node]['feature'])
                feature_dim = len(original_feature)
                
                # 随机选择要遮罩的维度
                n_mask = int(feature_dim * mask_ratio)
                mask_indices = random.sample(range(feature_dim), n_mask)
                
                # 创建遮罩后的特征（用0或均值替换）
                masked_feature = original_feature.copy()
                masked_feature[mask_indices] = 0  # 或使用 np.mean(original_feature)
                
                G_masked.nodes[node]['feature'] = masked_feature
                nodes_with_features += 1
                total_masked += n_mask
        
        print(f"  - 处理节点数: {nodes_with_features}")
        print(f"  - 总遮罩特征数: {total_masked}")
        
        return G_masked
    
    def apply_feature_generalization(self, n_bins: int = 10) -> nx.Graph:
        """
        特征泛化 - 将连续特征离散化为bins
        
        Args:
            n_bins: 离散化的bin数量
            
        Returns:
            特征泛化后的图
        """
        print(f"\n应用特征泛化 (bins: {n_bins})")
        
        G_generalized = self.G.copy()
        
        # 收集所有特征以计算全局bins
        all_features = []
        for node in G_generalized.nodes():
            if 'feature' in G_generalized.nodes[node]:
                all_features.append(G_generalized.nodes[node]['feature'])
        
        if not all_features:
            print("  - 警告: 没有节点特征")
            return G_generalized
        
        all_features = np.array(all_features)
        feature_dim = all_features.shape[1]
        
        # 对每个特征维度计算bins
        bins = []
        for d in range(feature_dim):
            feature_values = all_features[:, d]
            bin_edges = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
            bins.append(bin_edges)
        
        # 应用泛化
        nodes_processed = 0
        for node in G_generalized.nodes():
            if 'feature' in G_generalized.nodes[node]:
                original_feature = np.array(G_generalized.nodes[node]['feature'])
                generalized_feature = np.zeros_like(original_feature)
                
                for d in range(feature_dim):
                    # 找到所属的bin并用bin中心值替换
                    bin_idx = np.digitize(original_feature[d], bins[d]) - 1
                    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                    
                    # 使用bin的中心值
                    bin_center = (bins[d][bin_idx] + bins[d][bin_idx + 1]) / 2
                    generalized_feature[d] = bin_center
                
                G_generalized.nodes[node]['feature'] = generalized_feature
                nodes_processed += 1
        
        print(f"  - 处理节点数: {nodes_processed}")
        
        return G_generalized
    
    def calculate_feature_utility(self, G_original: nx.Graph, G_perturbed: nx.Graph) -> Dict:
        """
        计算特征效用保持
        
        Returns:
            效用指标字典
        """
        utilities = {}
        
        # 收集特征
        original_features = []
        perturbed_features = []
        
        for node in G_original.nodes():
            if 'feature' in G_original.nodes[node] and 'feature' in G_perturbed.nodes[node]:
                original_features.append(G_original.nodes[node]['feature'])
                perturbed_features.append(G_perturbed.nodes[node]['feature'])
        
        if not original_features:
            return {'error': 'No features found'}
        
        original_features = np.array(original_features)
        perturbed_features = np.array(perturbed_features)
        
        # 1. 欧氏距离
        euclidean_distances = np.linalg.norm(original_features - perturbed_features, axis=1)
        utilities['mean_euclidean_distance'] = float(np.mean(euclidean_distances))
        utilities['std_euclidean_distance'] = float(np.std(euclidean_distances))
        
        # 2. 余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sims = []
        for i in range(len(original_features)):
            sim = cosine_similarity(
                original_features[i].reshape(1, -1),
                perturbed_features[i].reshape(1, -1)
            )[0][0]
            cosine_sims.append(sim)
        
        utilities['mean_cosine_similarity'] = float(np.mean(cosine_sims))
        utilities['std_cosine_similarity'] = float(np.std(cosine_sims))
        
        # 3. 相对误差
        relative_errors = np.abs(original_features - perturbed_features) / (np.abs(original_features) + 1e-10)
        utilities['mean_relative_error'] = float(np.mean(relative_errors))
        
        return utilities


def test_feature_perturbation():
    """测试特征扰动防御"""
    print("\n" + "="*60)
    print("测试特征扰动防御模块")
    print("="*60)
    
    # 创建测试图（带有特征）
    G = nx.karate_club_graph()
    
    # 添加随机特征
    feature_dim = 10
    for node in G.nodes():
        G.nodes[node]['feature'] = np.random.rand(feature_dim)
    
    print(f"\n原始图: {G.number_of_nodes()} 节点, 特征维度: {feature_dim}")
    
    # 测试不同的扰动方法
    defense = FeaturePerturbationDefense(G, noise_level=0.1)
    
    # 1. 高斯噪声
    print("\n" + "="*60)
    print("方法 1: 高斯噪声")
    print("="*60)
    G_gaussian = defense.apply_gaussian_noise(seed=42)
    utilities_gaussian = defense.calculate_feature_utility(G, G_gaussian)
    print(f"平均欧氏距离: {utilities_gaussian['mean_euclidean_distance']:.4f}")
    print(f"平均余弦相似度: {utilities_gaussian['mean_cosine_similarity']:.4f}")
    
    # 2. 特征遮罩
    print("\n" + "="*60)
    print("方法 2: 特征遮罩")
    print("="*60)
    G_masked = defense.apply_feature_masking(mask_ratio=0.3, seed=42)
    utilities_masked = defense.calculate_feature_utility(G, G_masked)
    print(f"平均欧氏距离: {utilities_masked['mean_euclidean_distance']:.4f}")
    print(f"平均余弦相似度: {utilities_masked['mean_cosine_similarity']:.4f}")
    
    # 3. 特征泛化
    print("\n" + "="*60)
    print("方法 3: 特征泛化")
    print("="*60)
    G_generalized = defense.apply_feature_generalization(n_bins=5)
    utilities_gen = defense.calculate_feature_utility(G, G_generalized)
    print(f"平均欧氏距离: {utilities_gen['mean_euclidean_distance']:.4f}")
    print(f"平均余弦相似度: {utilities_gen['mean_cosine_similarity']:.4f}")


if __name__ == "__main__":
    test_feature_perturbation()





















