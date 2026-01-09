"""
图重构防御 - 通过重新生成图来保护隐私
实现基于图属性的重构算法
"""

import numpy as np
import networkx as nx
from typing import Dict, List
import random


class GraphReconstructionDefense:
    """图重构防御器"""
    
    def __init__(self, G: nx.Graph, preserve_properties: List[str] = None):
        """
        初始化图重构防御器
        
        Args:
            G: 原始图
            preserve_properties: 要保留的图属性列表
                ['degree_distribution', 'clustering', 'community', 'diameter']
        """
        self.G = G.copy()
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()
        self.preserve_properties = preserve_properties or ['degree_distribution']
    
    def reconstruct_with_configuration_model(self, seed: int = None) -> nx.Graph:
        """
        使用配置模型重构图（保持度序列）
        
        Args:
            seed: 随机种子
            
        Returns:
            重构后的图
        """
        if seed is not None:
            random.seed(seed)
        
        print(f"\n使用配置模型重构图")
        print(f"  - 原始节点数: {self.n_nodes}")
        print(f"  - 原始边数: {self.n_edges}")
        
        # 获取度序列
        degree_sequence = [d for n, d in self.G.degree()]
        
        # 如果度序列总和为奇数，调整一个节点的度数
        if sum(degree_sequence) % 2 == 1:
            degree_sequence[0] += 1
        
        try:
            # 使用配置模型生成新图
            G_reconstructed = nx.configuration_model(degree_sequence, seed=seed)
            
            # 移除自环和多重边
            G_reconstructed = nx.Graph(G_reconstructed)
            G_reconstructed.remove_edges_from(nx.selfloop_edges(G_reconstructed))
            
            # 重新映射节点ID以匹配原图
            mapping = dict(zip(G_reconstructed.nodes(), self.G.nodes()))
            G_reconstructed = nx.relabel_nodes(G_reconstructed, mapping)
            
            print(f"  - 重构后节点数: {G_reconstructed.number_of_nodes()}")
            print(f"  - 重构后边数: {G_reconstructed.number_of_edges()}")
            
            return G_reconstructed
        
        except Exception as e:
            print(f"  ✗ 配置模型失败: {e}")
            print(f"  返回原图副本")
            return self.G.copy()
    
    def reconstruct_with_stochastic_block_model(self, n_communities: int = 5, seed: int = None) -> nx.Graph:
        """
        使用随机块模型重构图（保持社区结构）
        
        Args:
            n_communities: 社区数量
            seed: 随机种子
            
        Returns:
            重构后的图
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"\n使用随机块模型重构图 (社区数: {n_communities})")
        
        try:
            # 尝试检测原图的社区
            from community import community_louvain
            communities = community_louvain.best_partition(self.G)
            
            # 统计每个社区的节点数
            community_sizes = {}
            for node, comm in communities.items():
                community_sizes[comm] = community_sizes.get(comm, 0) + 1
            
            # 构建社区大小列表
            sizes = [community_sizes.get(i, self.n_nodes // n_communities) 
                    for i in range(n_communities)]
            
            # 确保总节点数匹配
            while sum(sizes) < self.n_nodes:
                sizes[0] += 1
            while sum(sizes) > self.n_nodes:
                sizes[-1] -= 1
            
            # 估计社区间和社区内的边概率
            p_in = self.n_edges / (self.n_nodes * (self.n_nodes - 1) / 2) * 3  # 社区内
            p_out = p_in * 0.1  # 社区间
            
            # 构建概率矩阵
            probs = np.full((n_communities, n_communities), p_out)
            np.fill_diagonal(probs, p_in)
            
            # 生成随机块模型图
            G_reconstructed = nx.stochastic_block_model(sizes, probs, seed=seed)
            
            # 重新映射节点ID
            mapping = dict(zip(G_reconstructed.nodes(), self.G.nodes()))
            G_reconstructed = nx.relabel_nodes(G_reconstructed, mapping)
            
            print(f"  - 重构后节点数: {G_reconstructed.number_of_nodes()}")
            print(f"  - 重构后边数: {G_reconstructed.number_of_edges()}")
            
            return G_reconstructed
        
        except Exception as e:
            print(f"  ✗ 随机块模型失败: {e}")
            print(f"  回退到配置模型")
            return self.reconstruct_with_configuration_model(seed=seed)
    
    def reconstruct_with_random_graph(self, model: str = 'erdos_renyi', seed: int = None) -> nx.Graph:
        """
        使用随机图模型重构
        
        Args:
            model: 模型类型 ('erdos_renyi', 'barabasi_albert', 'watts_strogatz')
            seed: 随机种子
            
        Returns:
            重构后的图
        """
        if seed is not None:
            random.seed(seed)
        
        print(f"\n使用随机图模型重构 (模型: {model})")
        
        # 计算边概率
        p = self.n_edges / (self.n_nodes * (self.n_nodes - 1) / 2)
        
        if model == 'erdos_renyi':
            G_reconstructed = nx.erdos_renyi_graph(self.n_nodes, p, seed=seed)
        
        elif model == 'barabasi_albert':
            m = max(1, self.n_edges // self.n_nodes)  # 每次添加的边数
            G_reconstructed = nx.barabasi_albert_graph(self.n_nodes, m, seed=seed)
        
        elif model == 'watts_strogatz':
            k = max(2, int(2 * self.n_edges / self.n_nodes))  # 平均度
            if k % 2 == 1:
                k += 1  # k必须是偶数
            k = min(k, self.n_nodes - 1)
            G_reconstructed = nx.watts_strogatz_graph(self.n_nodes, k, 0.1, seed=seed)
        
        else:
            print(f"  ✗ 未知模型: {model}")
            return self.G.copy()
        
        # 重新映射节点ID
        mapping = dict(zip(G_reconstructed.nodes(), self.G.nodes()))
        G_reconstructed = nx.relabel_nodes(G_reconstructed, mapping)
        
        print(f"  - 重构后节点数: {G_reconstructed.number_of_nodes()}")
        print(f"  - 重构后边数: {G_reconstructed.number_of_edges()}")
        
        return G_reconstructed
    
    def calculate_structural_similarity(self, G_reconstructed: nx.Graph) -> Dict:
        """
        计算重构图与原图的结构相似度
        
        Returns:
            相似度指标字典
        """
        similarities = {}
        
        # 1. 度分布相似度（KL散度）
        from scipy.stats import entropy
        
        deg_orig = dict(self.G.degree())
        deg_recon = dict(G_reconstructed.degree())
        
        max_degree = max(max(deg_orig.values()), max(deg_recon.values()))
        
        hist_orig = np.zeros(max_degree + 1)
        hist_recon = np.zeros(max_degree + 1)
        
        for d in deg_orig.values():
            hist_orig[d] += 1
        for d in deg_recon.values():
            hist_recon[d] += 1
        
        hist_orig = hist_orig / hist_orig.sum()
        hist_recon = hist_recon / hist_recon.sum()
        
        # 避免零值
        hist_orig = hist_orig + 1e-10
        hist_recon = hist_recon + 1e-10
        
        kl_div = entropy(hist_orig, hist_recon)
        similarities['degree_distribution_kl'] = float(kl_div)
        
        # 2. 平均度相似度
        avg_deg_orig = 2 * self.n_edges / self.n_nodes
        avg_deg_recon = 2 * G_reconstructed.number_of_edges() / G_reconstructed.number_of_nodes()
        similarities['avg_degree_diff'] = abs(avg_deg_orig - avg_deg_recon)
        
        # 3. 聚类系数相似度
        try:
            clustering_orig = nx.average_clustering(self.G)
            clustering_recon = nx.average_clustering(G_reconstructed)
            similarities['clustering_diff'] = abs(clustering_orig - clustering_recon)
        except:
            similarities['clustering_diff'] = None
        
        # 4. 边重叠
        edges_orig = set(self.G.edges())
        edges_recon = set(G_reconstructed.edges())
        
        overlap = len(edges_orig & edges_recon)
        similarities['edge_overlap'] = overlap / len(edges_orig) if edges_orig else 0
        
        return similarities


def test_graph_reconstruction():
    """测试图重构防御"""
    print("\n" + "="*60)
    print("测试图重构防御模块")
    print("="*60)
    
    # 创建测试图
    G = nx.karate_club_graph()
    print(f"\n原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    defense = GraphReconstructionDefense(G)
    
    # 测试不同的重构方法
    methods = [
        ('配置模型', lambda: defense.reconstruct_with_configuration_model(seed=42)),
        ('随机块模型', lambda: defense.reconstruct_with_stochastic_block_model(n_communities=3, seed=42)),
        ('ER随机图', lambda: defense.reconstruct_with_random_graph(model='erdos_renyi', seed=42)),
    ]
    
    for method_name, method_func in methods:
        print(f"\n{'='*60}")
        print(f"方法: {method_name}")
        print(f"{'='*60}")
        
        G_reconstructed = method_func()
        similarities = defense.calculate_structural_similarity(G_reconstructed)
        
        print(f"\n结构相似度:")
        print(f"  - 度分布KL散度: {similarities['degree_distribution_kl']:.4f}")
        print(f"  - 平均度差异: {similarities['avg_degree_diff']:.2f}")
        if similarities['clustering_diff'] is not None:
            print(f"  - 聚类系数差异: {similarities['clustering_diff']:.4f}")
        print(f"  - 边重叠率: {similarities['edge_overlap']:.2%}")


if __name__ == "__main__":
    test_graph_reconstruction()





















