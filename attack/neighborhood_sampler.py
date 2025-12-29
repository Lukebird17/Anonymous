"""
邻域采样模块 - 模拟攻击者只能获取局部子图的场景
用于第二阶段：现实场景模拟
"""

import networkx as nx
import numpy as np
from typing import List, Set, Dict, Tuple
import random


class NeighborhoodSampler:
    """二阶邻域采样器 - 模拟碎片化信息场景"""
    
    def __init__(self, G: nx.Graph):
        """
        初始化采样器
        
        Args:
            G: 原始图
        """
        self.G = G
        
    def sample_k_hop_neighbors(self, target_node: int, k: int = 2) -> nx.Graph:
        """
        采样目标节点的k阶邻居子图
        
        Args:
            target_node: 目标节点
            k: 邻居阶数（默认2阶）
            
        Returns:
            k阶邻居子图
        """
        if target_node not in self.G:
            raise ValueError(f"节点 {target_node} 不在图中")
        
        # BFS获取k阶邻居
        neighbors = self._get_k_hop_neighbors(target_node, k)
        neighbors.add(target_node)
        
        # 提取子图
        subgraph = self.G.subgraph(neighbors).copy()
        
        return subgraph
    
    def _get_k_hop_neighbors(self, node: int, k: int) -> Set[int]:
        """获取k阶邻居节点集合"""
        if k == 0:
            return set()
        
        neighbors = set()
        current_level = {node}
        
        for _ in range(k):
            next_level = set()
            for n in current_level:
                if n in self.G:
                    next_level.update(self.G.neighbors(n))
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors
    
    def sample_multiple_targets(self, target_nodes: List[int], k: int = 2) -> Dict[int, nx.Graph]:
        """
        为多个目标节点采样k阶邻居子图
        
        Args:
            target_nodes: 目标节点列表
            k: 邻居阶数
            
        Returns:
            {节点ID: 子图} 字典
        """
        subgraphs = {}
        for node in target_nodes:
            try:
                subgraphs[node] = self.sample_k_hop_neighbors(node, k)
            except ValueError:
                print(f"警告: 节点 {node} 不在图中，跳过")
        
        return subgraphs
    
    def random_sample_targets(self, n_targets: int, k: int = 2, 
                            min_degree: int = 5) -> Dict[int, nx.Graph]:
        """
        随机选择n个目标节点并采样其k阶邻居
        
        Args:
            n_targets: 目标节点数量
            k: 邻居阶数
            min_degree: 最小度数要求（过滤孤立节点）
            
        Returns:
            {节点ID: 子图} 字典
        """
        # 过滤度数太小的节点
        candidate_nodes = [node for node in self.G.nodes() 
                          if self.G.degree(node) >= min_degree]
        
        if len(candidate_nodes) < n_targets:
            print(f"警告: 符合条件的节点数 ({len(candidate_nodes)}) < 目标数量 ({n_targets})")
            n_targets = len(candidate_nodes)
        
        # 随机选择目标节点
        selected_nodes = random.sample(candidate_nodes, n_targets)
        
        return self.sample_multiple_targets(selected_nodes, k)
    
    def get_subgraph_statistics(self, subgraph: nx.Graph) -> Dict:
        """
        计算子图统计信息
        
        Args:
            subgraph: 子图
            
        Returns:
            统计信息字典
        """
        stats = {
            'n_nodes': subgraph.number_of_nodes(),
            'n_edges': subgraph.number_of_edges(),
            'avg_degree': 2 * subgraph.number_of_edges() / subgraph.number_of_nodes() if subgraph.number_of_nodes() > 0 else 0,
            'density': nx.density(subgraph),
            'n_components': nx.number_connected_components(subgraph)
        }
        
        return stats


class RobustnessSimulator:
    """鲁棒性测试器 - 模拟边的随机缺失"""
    
    def __init__(self, G: nx.Graph):
        """
        初始化鲁棒性测试器
        
        Args:
            G: 原始图
        """
        self.G = G.copy()
        self.original_edges = list(self.G.edges())
    
    def drop_edges_random(self, drop_ratio: float) -> nx.Graph:
        """
        随机删除一定比例的边
        
        Args:
            drop_ratio: 删除边的比例 (0.0 ~ 1.0)
            
        Returns:
            删除边后的图
        """
        G_incomplete = self.G.copy()
        
        # 随机选择要删除的边
        n_edges_to_drop = int(len(self.original_edges) * drop_ratio)
        edges_to_drop = random.sample(self.original_edges, n_edges_to_drop)
        
        # 删除边
        G_incomplete.remove_edges_from(edges_to_drop)
        
        return G_incomplete
    
    def generate_incomplete_graphs(self, drop_ratios: List[float]) -> Dict[float, nx.Graph]:
        """
        生成不同完整度的图
        
        Args:
            drop_ratios: 删除边的比例列表，如 [0.1, 0.2, 0.3, 0.4, 0.5]
            
        Returns:
            {删除比例: 图} 字典
        """
        incomplete_graphs = {}
        
        for ratio in drop_ratios:
            G_incomplete = self.drop_edges_random(ratio)
            incomplete_graphs[ratio] = G_incomplete
            print(f"生成 {ratio*100:.0f}% 边缺失的图: "
                  f"{G_incomplete.number_of_nodes()} 节点, "
                  f"{G_incomplete.number_of_edges()} 边")
        
        return incomplete_graphs
    
    def calculate_completeness(self, G_incomplete: nx.Graph) -> float:
        """
        计算图的完整度
        
        Args:
            G_incomplete: 不完整的图
            
        Returns:
            完整度 (0.0 ~ 1.0)
        """
        original_edges = self.G.number_of_edges()
        current_edges = G_incomplete.number_of_edges()
        
        return current_edges / original_edges if original_edges > 0 else 0.0


class LocalViewGenerator:
    """局部视图生成器 - 综合邻域采样和边缺失"""
    
    def __init__(self, G: nx.Graph):
        """
        初始化局部视图生成器
        
        Args:
            G: 原始完整图
        """
        self.G = G
        self.sampler = NeighborhoodSampler(G)
        self.robustness = RobustnessSimulator(G)
    
    def generate_local_view(self, target_node: int, k: int = 2, 
                           edge_drop_ratio: float = 0.0) -> nx.Graph:
        """
        生成目标节点的局部视图（k阶邻居 + 边缺失）
        
        Args:
            target_node: 目标节点
            k: 邻居阶数
            edge_drop_ratio: 边缺失比例
            
        Returns:
            局部视图子图
        """
        # 1. 采样k阶邻居
        subgraph = self.sampler.sample_k_hop_neighbors(target_node, k)
        
        # 2. 如果需要，随机删除一些边
        if edge_drop_ratio > 0:
            edges = list(subgraph.edges())
            n_drop = int(len(edges) * edge_drop_ratio)
            if n_drop > 0:
                edges_to_drop = random.sample(edges, n_drop)
                subgraph.remove_edges_from(edges_to_drop)
        
        return subgraph
    
    def batch_generate_local_views(self, n_targets: int, k: int = 2,
                                   edge_drop_ratios: List[float] = [0.0, 0.1, 0.2, 0.3]) -> Dict:
        """
        批量生成不同完整度的局部视图
        
        Args:
            n_targets: 目标节点数量
            k: 邻居阶数
            edge_drop_ratios: 边缺失比例列表
            
        Returns:
            嵌套字典: {drop_ratio: {target_node: subgraph}}
        """
        # 随机选择目标节点
        candidate_nodes = [n for n in self.G.nodes() if self.G.degree(n) >= 3]
        if len(candidate_nodes) < n_targets:
            n_targets = len(candidate_nodes)
        target_nodes = random.sample(candidate_nodes, n_targets)
        
        # 为每个drop_ratio生成局部视图
        results = {}
        for drop_ratio in edge_drop_ratios:
            results[drop_ratio] = {}
            for node in target_nodes:
                subgraph = self.generate_local_view(node, k, drop_ratio)
                results[drop_ratio][node] = subgraph
            
            print(f"已生成 {len(target_nodes)} 个局部视图 (边缺失: {drop_ratio*100:.0f}%)")
        
        return results


def test_neighborhood_sampling():
    """测试邻域采样功能"""
    print("\n" + "="*60)
    print("测试邻域采样模块")
    print("="*60)
    
    # 创建测试图
    G = nx.karate_club_graph()
    print(f"\n使用空手道俱乐部图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")
    
    # 测试1: 单节点k阶邻居采样
    print("【测试1】单节点2阶邻居采样")
    print("-" * 60)
    sampler = NeighborhoodSampler(G)
    target = 0
    subgraph = sampler.sample_k_hop_neighbors(target, k=2)
    stats = sampler.get_subgraph_statistics(subgraph)
    print(f"目标节点: {target}")
    print(f"子图规模: {stats['n_nodes']} 节点, {stats['n_edges']} 边")
    print(f"平均度: {stats['avg_degree']:.2f}")
    print(f"密度: {stats['density']:.4f}")
    
    # 测试2: 批量采样
    print("\n【测试2】批量随机采样 (5个目标节点)")
    print("-" * 60)
    subgraphs = sampler.random_sample_targets(n_targets=5, k=2, min_degree=5)
    for node, sg in subgraphs.items():
        print(f"节点 {node}: {sg.number_of_nodes()} 节点, {sg.number_of_edges()} 边")
    
    # 测试3: 鲁棒性测试
    print("\n【测试3】鲁棒性测试 - 逐步删除边")
    print("-" * 60)
    robustness = RobustnessSimulator(G)
    drop_ratios = [0.1, 0.2, 0.3, 0.5]
    incomplete_graphs = robustness.generate_incomplete_graphs(drop_ratios)
    
    # 测试4: 局部视图生成
    print("\n【测试4】局部视图生成 (综合测试)")
    print("-" * 60)
    view_gen = LocalViewGenerator(G)
    local_views = view_gen.batch_generate_local_views(
        n_targets=3, 
        k=2, 
        edge_drop_ratios=[0.0, 0.2, 0.4]
    )
    
    print("\n局部视图统计:")
    for drop_ratio, views in local_views.items():
        avg_nodes = np.mean([v.number_of_nodes() for v in views.values()])
        avg_edges = np.mean([v.number_of_edges() for v in views.values()])
        print(f"  边缺失 {drop_ratio*100:.0f}%: 平均 {avg_nodes:.1f} 节点, {avg_edges:.1f} 边")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    test_neighborhood_sampling()

