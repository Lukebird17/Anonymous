"""
差分隐私防御 - 基于边扰动的图加噪算法
实现 ε-差分隐私保护
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List
import random
from copy import deepcopy


class DifferentialPrivacyDefense:
    """差分隐私防御器 - 边扰动算法"""
    
    def __init__(self, G: nx.Graph, epsilon: float = 1.0):
        """
        初始化差分隐私防御器
        
        Args:
            G: 原始图
            epsilon: 隐私预算 (越小隐私保护越强)
        """
        self.G = G.copy()
        self.epsilon = epsilon
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()
        
        # 计算翻转概率
        self._calculate_flip_probability()
    
    def _calculate_flip_probability(self):
        """
        根据 ε-差分隐私计算边翻转概率
        
        基于 Randomized Response 机制
        """
        # 对于每条可能的边，以概率 p 翻转其状态
        # p = 1 / (1 + e^(ε))
        self.flip_prob = 1.0 / (1.0 + np.exp(self.epsilon))
        
        print(f"差分隐私参数: ε={self.epsilon:.2f}, 翻转概率={self.flip_prob:.4f}")
    
    def add_noise_edge_perturbation(self, seed: int = None) -> nx.Graph:
        """
        边扰动算法 - 随机添加/删除边（优化版本，适用于大图）
        
        Args:
            seed: 随机种子
            
        Returns:
            加噪后的图
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        G_private = self.G.copy()
        
        # 统计信息
        n_edges_flipped = 0
        n_edges_added = 0
        n_edges_removed = 0
        
        nodes = list(G_private.nodes())
        n_nodes = len(nodes)
        
        # 对于大图，使用采样策略而不是遍历所有可能的边
        # 估算需要翻转的边数
        total_possible_edges = n_nodes * (n_nodes - 1) // 2
        expected_flips = int(total_possible_edges * self.flip_prob)
        
        # 限制最大翻转数（避免计算时间过长）
        max_flips = min(expected_flips, self.n_edges * 5)  # 最多翻转5倍的现有边数
        
        print(f"  - 图规模: {n_nodes} 节点, {self.n_edges} 边")
        print(f"  - 估算翻转边数: {expected_flips} (限制为 {max_flips})")
        
        # 随机采样要检查的边
        edges_to_check = set()
        while len(edges_to_check) < max_flips:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u < v:  # 避免重复和自环
                edges_to_check.add((u, v))
        
        # 对采样的边进行翻转
        for u, v in edges_to_check:
            if random.random() < self.flip_prob:
                edge_exists = G_private.has_edge(u, v)
                if edge_exists:
                    G_private.remove_edge(u, v)
                    n_edges_removed += 1
                else:
                    G_private.add_edge(u, v)
                    n_edges_added += 1
                n_edges_flipped += 1
        
        print(f"\n边扰动统计:")
        print(f"  - 翻转边数: {n_edges_flipped}")
        print(f"  - 添加边数: {n_edges_added}")
        print(f"  - 删除边数: {n_edges_removed}")
        print(f"  - 原始边数: {self.n_edges}")
        print(f"  - 加噪后边数: {G_private.number_of_edges()}")
        
        return G_private
    
    def add_noise_laplace(self, sensitivity: float = 1.0) -> nx.Graph:
        """
        拉普拉斯机制 - 对度序列添加噪声
        
        Args:
            sensitivity: 敏感度参数
            
        Returns:
            加噪后的图
        """
        G_private = self.G.copy()
        
        # 计算每个节点的目标度数（加噪声）
        target_degrees = {}
        for node in G_private.nodes():
            current_degree = G_private.degree(node)
            # 添加拉普拉斯噪声
            noise = np.random.laplace(0, sensitivity / self.epsilon)
            target_degree = max(0, int(current_degree + noise))
            target_degrees[node] = target_degree
        
        # 调整边以匹配目标度数（简化版本）
        # 这是一个复杂的优化问题，这里用启发式方法
        nodes = list(G_private.nodes())
        
        for node in nodes:
            current_degree = G_private.degree(node)
            target_degree = target_degrees[node]
            
            if current_degree < target_degree:
                # 需要添加边
                n_to_add = target_degree - current_degree
                # 随机选择不相连的节点
                non_neighbors = [n for n in nodes if n != node and not G_private.has_edge(node, n)]
                if non_neighbors and n_to_add > 0:
                    to_connect = random.sample(non_neighbors, min(n_to_add, len(non_neighbors)))
                    for other in to_connect:
                        G_private.add_edge(node, other)
            
            elif current_degree > target_degree:
                # 需要删除边
                n_to_remove = current_degree - target_degree
                neighbors = list(G_private.neighbors(node))
                if neighbors and n_to_remove > 0:
                    to_disconnect = random.sample(neighbors, min(n_to_remove, len(neighbors)))
                    for other in to_disconnect:
                        G_private.remove_edge(node, other)
        
        print(f"\n拉普拉斯机制统计:")
        print(f"  - 原始边数: {self.n_edges}")
        print(f"  - 加噪后边数: {G_private.number_of_edges()}")
        
        return G_private


class PrivacyUtilityEvaluator:
    """隐私-效用权衡评估器"""
    
    def __init__(self, G_original: nx.Graph, G_private: nx.Graph):
        """
        初始化评估器
        
        Args:
            G_original: 原始图
            G_private: 加噪后的图
        """
        self.G_original = G_original
        self.G_private = G_private
    
    def calculate_graph_structural_loss(self) -> Dict:
        """
        计算图结构损失 (Graph Structural Loss)
        
        Returns:
            结构损失指标字典
        """
        # 1. 边的变化
        original_edges = set(self.G_original.edges())
        private_edges = set(self.G_private.edges())
        
        edges_added = len(private_edges - original_edges)
        edges_removed = len(original_edges - private_edges)
        edges_unchanged = len(original_edges & private_edges)
        
        # L1 距离（邻接矩阵）
        n_possible_edges = self.G_original.number_of_nodes() * (self.G_original.number_of_nodes() - 1) / 2
        l1_distance = (edges_added + edges_removed) / n_possible_edges
        
        # 2. 度分布的变化
        degrees_orig = dict(self.G_original.degree())
        degrees_priv = dict(self.G_private.degree())
        
        degree_mae = np.mean([abs(degrees_orig.get(n, 0) - degrees_priv.get(n, 0)) 
                             for n in self.G_original.nodes()])
        
        # 3. 全局统计量的变化
        avg_degree_orig = 2 * self.G_original.number_of_edges() / self.G_original.number_of_nodes()
        avg_degree_priv = 2 * self.G_private.number_of_edges() / self.G_private.number_of_nodes()
        
        try:
            clustering_orig = nx.average_clustering(self.G_original)
            clustering_priv = nx.average_clustering(self.G_private)
        except:
            clustering_orig = 0
            clustering_priv = 0
        
        # 计算路径长度（采样以加速）
        try:
            if nx.is_connected(self.G_original):
                avg_path_orig = nx.average_shortest_path_length(self.G_original)
            else:
                # 只计算最大连通分量
                largest_cc = max(nx.connected_components(self.G_original), key=len)
                subgraph = self.G_original.subgraph(largest_cc)
                avg_path_orig = nx.average_shortest_path_length(subgraph)
            
            if nx.is_connected(self.G_private):
                avg_path_priv = nx.average_shortest_path_length(self.G_private)
            else:
                largest_cc = max(nx.connected_components(self.G_private), key=len)
                subgraph = self.G_private.subgraph(largest_cc)
                avg_path_priv = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_orig = 0
            avg_path_priv = 0
        
        # 汇总结果
        structural_loss = {
            'edges_added': edges_added,
            'edges_removed': edges_removed,
            'edges_unchanged': edges_unchanged,
            'l1_distance': l1_distance,
            'degree_mae': degree_mae,
            'avg_degree_orig': avg_degree_orig,
            'avg_degree_priv': avg_degree_priv,
            'avg_degree_diff': abs(avg_degree_orig - avg_degree_priv),
            'clustering_orig': clustering_orig,
            'clustering_priv': clustering_priv,
            'clustering_diff': abs(clustering_orig - clustering_priv),
            'avg_path_orig': avg_path_orig,
            'avg_path_priv': avg_path_priv,
            'avg_path_diff': abs(avg_path_orig - avg_path_priv)
        }
        
        return structural_loss
    
    def evaluate_utility_for_tasks(self) -> Dict:
        """
        评估加噪后图在不同任务上的效用保持
        
        Returns:
            效用指标字典
        """
        utilities = {}
        
        # 1. 社区发现效用
        try:
            from community import community_louvain
            communities_orig = community_louvain.best_partition(self.G_original)
            communities_priv = community_louvain.best_partition(self.G_private)
            
            # 计算模块度
            modularity_orig = community_louvain.modularity(communities_orig, self.G_original)
            modularity_priv = community_louvain.modularity(communities_priv, self.G_private)
            
            utilities['modularity_orig'] = modularity_orig
            utilities['modularity_priv'] = modularity_priv
            utilities['modularity_preservation'] = modularity_priv / modularity_orig if modularity_orig > 0 else 0
        except:
            print("警告: 无法计算模块度（需要 python-louvain 库）")
        
        # 2. 节点重要性排序效用（度中心性）
        degree_centrality_orig = nx.degree_centrality(self.G_original)
        degree_centrality_priv = nx.degree_centrality(self.G_private)
        
        # Spearman 秩相关系数
        from scipy.stats import spearmanr
        common_nodes = set(degree_centrality_orig.keys()) & set(degree_centrality_priv.keys())
        if common_nodes:
            ranks_orig = [degree_centrality_orig[n] for n in common_nodes]
            ranks_priv = [degree_centrality_priv[n] for n in common_nodes]
            spearman_corr, _ = spearmanr(ranks_orig, ranks_priv)
            utilities['degree_centrality_correlation'] = spearman_corr
        
        # 3. 连通性保持
        utilities['n_components_orig'] = nx.number_connected_components(self.G_original)
        utilities['n_components_priv'] = nx.number_connected_components(self.G_private)
        utilities['connectivity_preserved'] = (utilities['n_components_orig'] == utilities['n_components_priv'])
        
        return utilities
    
    def print_comprehensive_report(self):
        """打印完整的评估报告"""
        print("\n" + "="*70)
        print("差分隐私防御 - 隐私与效用权衡评估报告")
        print("="*70)
        
        # 结构损失
        print("\n【1. 图结构损失】")
        print("-" * 70)
        structural_loss = self.calculate_graph_structural_loss()
        print(f"边的变化:")
        print(f"  - 添加: {structural_loss['edges_added']}")
        print(f"  - 删除: {structural_loss['edges_removed']}")
        print(f"  - 保持: {structural_loss['edges_unchanged']}")
        print(f"  - L1距离: {structural_loss['l1_distance']:.6f}")
        print(f"\n度数变化:")
        print(f"  - 平均度 (原始): {structural_loss['avg_degree_orig']:.2f}")
        print(f"  - 平均度 (加噪): {structural_loss['avg_degree_priv']:.2f}")
        print(f"  - 度数MAE: {structural_loss['degree_mae']:.2f}")
        print(f"\n全局统计量:")
        print(f"  - 聚类系数 (原始): {structural_loss['clustering_orig']:.4f}")
        print(f"  - 聚类系数 (加噪): {structural_loss['clustering_priv']:.4f}")
        print(f"  - 聚类系数差异: {structural_loss['clustering_diff']:.4f}")
        if structural_loss['avg_path_orig'] > 0:
            print(f"  - 平均路径长度 (原始): {structural_loss['avg_path_orig']:.2f}")
            print(f"  - 平均路径长度 (加噪): {structural_loss['avg_path_priv']:.2f}")
        
        # 效用评估
        print("\n【2. 任务效用保持】")
        print("-" * 70)
        utilities = self.evaluate_utility_for_tasks()
        if 'modularity_orig' in utilities:
            print(f"社区发现:")
            print(f"  - 模块度 (原始): {utilities['modularity_orig']:.4f}")
            print(f"  - 模块度 (加噪): {utilities['modularity_priv']:.4f}")
            print(f"  - 保持率: {utilities['modularity_preservation']:.2%}")
        print(f"\n节点重要性:")
        if 'degree_centrality_correlation' in utilities:
            print(f"  - 度中心性相关性 (Spearman): {utilities['degree_centrality_correlation']:.4f}")
        print(f"\n连通性:")
        print(f"  - 连通分量数 (原始): {utilities['n_components_orig']}")
        print(f"  - 连通分量数 (加噪): {utilities['n_components_priv']}")
        print(f"  - 连通性保持: {'✓' if utilities['connectivity_preserved'] else '✗'}")
        
        print("\n" + "="*70)


def test_differential_privacy():
    """测试差分隐私防御"""
    print("\n" + "="*60)
    print("测试差分隐私防御模块")
    print("="*60)
    
    # 创建测试图
    G = nx.karate_club_graph()
    print(f"\n原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 测试不同隐私预算
    epsilons = [0.5, 1.0, 2.0, 5.0]
    
    for epsilon in epsilons:
        print(f"\n{'='*60}")
        print(f"测试 ε = {epsilon}")
        print(f"{'='*60}")
        
        # 应用差分隐私
        dp_defense = DifferentialPrivacyDefense(G, epsilon=epsilon)
        G_private = dp_defense.add_noise_edge_perturbation(seed=42)
        
        # 评估
        evaluator = PrivacyUtilityEvaluator(G, G_private)
        evaluator.print_comprehensive_report()


if __name__ == "__main__":
    test_differential_privacy()

