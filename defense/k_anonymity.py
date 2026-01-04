"""
K-匿名性防御 - 基于度序列的K-匿名保护
实现图的K-匿名性保护，使得每个节点至少与k-1个其他节点具有相同的度
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from collections import Counter
import random


class KAnonymityDefense:
    """K-匿名性防御器"""
    
    def __init__(self, G: nx.Graph, k: int = 2):
        """
        初始化K-匿名性防御器
        
        Args:
            G: 原始图
            k: 匿名参数（每个度值至少有k个节点）
        """
        self.G = G.copy()
        self.k = k
        self.n_nodes = G.number_of_nodes()
        self.n_edges = G.number_of_edges()
    
    def apply_k_anonymity(self, method='add_edges') -> nx.Graph:
        """
        应用K-匿名性保护
        
        Args:
            method: 保护方法 ('add_edges', 'remove_edges', 'both')
            
        Returns:
            K-匿名保护后的图
        """
        print(f"\n应用 {self.k}-匿名性保护 (方法: {method})")
        
        G_protected = self.G.copy()
        
        # 1. 统计当前度分布
        degree_sequence = dict(G_protected.degree())
        degree_counts = Counter(degree_sequence.values())
        
        print(f"原始度分布: {len(degree_counts)} 种不同的度值")
        
        # 2. 识别不满足K-匿名性的节点
        nodes_to_adjust = []
        for node, degree in degree_sequence.items():
            if degree_counts[degree] < self.k:
                nodes_to_adjust.append(node)
        
        print(f"需要调整的节点: {len(nodes_to_adjust)} / {self.n_nodes}")
        
        if not nodes_to_adjust:
            print("✓ 图已满足K-匿名性")
            return G_protected
        
        # 3. 调整节点度数以满足K-匿名性
        if method in ['add_edges', 'both']:
            G_protected = self._add_edges_for_k_anonymity(G_protected, nodes_to_adjust)
        
        if method in ['remove_edges', 'both']:
            G_protected = self._remove_edges_for_k_anonymity(G_protected, nodes_to_adjust)
        
        # 4. 验证K-匿名性
        final_degree_counts = Counter(dict(G_protected.degree()).values())
        violations = sum(1 for count in final_degree_counts.values() if count < self.k)
        
        print(f"\n调整后统计:")
        print(f"  - 原始边数: {self.n_edges}")
        print(f"  - 调整后边数: {G_protected.number_of_edges()}")
        print(f"  - 边数变化: {G_protected.number_of_edges() - self.n_edges:+d}")
        print(f"  - K-匿名性违反数: {violations}")
        
        return G_protected
    
    def _add_edges_for_k_anonymity(self, G: nx.Graph, nodes_to_adjust: List) -> nx.Graph:
        """通过添加边来满足K-匿名性"""
        degree_sequence = dict(G.degree())
        degree_counts = Counter(degree_sequence.values())
        
        # 对每个不满足K-匿名性的度值，找到最近的满足条件的目标度值
        for node in nodes_to_adjust:
            current_degree = degree_sequence[node]
            
            # 如果当前度值的节点数小于k，则增加度数到下一个满足k的度值
            if degree_counts[current_degree] < self.k:
                # 寻找更高的满足k-匿名的度值
                target_degree = current_degree
                for d in range(current_degree + 1, self.n_nodes):
                    if degree_counts[d] >= self.k:
                        target_degree = d
                        break
                else:
                    # 如果没找到，就增加到使得有k个节点的度值
                    target_degree = current_degree + 1
                
                # 添加边直到达到目标度数
                edges_to_add = target_degree - current_degree
                candidates = [n for n in G.nodes() 
                             if n != node and not G.has_edge(node, n)]
                
                if candidates and edges_to_add > 0:
                    selected = random.sample(candidates, min(edges_to_add, len(candidates)))
                    for other_node in selected:
                        G.add_edge(node, other_node)
                    
                    # 更新度数统计
                    degree_counts[current_degree] -= 1
                    new_degree = G.degree(node)
                    degree_counts[new_degree] = degree_counts.get(new_degree, 0) + 1
        
        return G
    
    def _remove_edges_for_k_anonymity(self, G: nx.Graph, nodes_to_adjust: List) -> nx.Graph:
        """通过删除边来满足K-匿名性（谨慎使用，会破坏图结构）"""
        degree_sequence = dict(G.degree())
        degree_counts = Counter(degree_sequence.values())
        
        for node in nodes_to_adjust:
            current_degree = degree_sequence[node]
            
            if degree_counts[current_degree] < self.k:
                # 寻找更低的满足k-匿名的度值
                target_degree = current_degree
                for d in range(current_degree - 1, -1, -1):
                    if degree_counts[d] >= self.k:
                        target_degree = d
                        break
                
                # 删除边直到达到目标度数
                edges_to_remove = current_degree - target_degree
                neighbors = list(G.neighbors(node))
                
                if neighbors and edges_to_remove > 0:
                    selected = random.sample(neighbors, min(edges_to_remove, len(neighbors)))
                    for other_node in selected:
                        G.remove_edge(node, other_node)
                    
                    # 更新度数统计
                    degree_counts[current_degree] -= 1
                    new_degree = G.degree(node)
                    degree_counts[new_degree] = degree_counts.get(new_degree, 0) + 1
        
        return G
    
    def calculate_anonymity_score(self, G: nx.Graph) -> float:
        """
        计算图的匿名性得分
        
        Returns:
            匿名性得分 (0-1, 越高越好)
        """
        degree_counts = Counter(dict(G.degree()).values())
        
        # 计算满足k-匿名的节点比例
        satisfied_nodes = sum(count for count in degree_counts.values() if count >= self.k)
        total_nodes = G.number_of_nodes()
        
        return satisfied_nodes / total_nodes if total_nodes > 0 else 0


def test_k_anonymity():
    """测试K-匿名性防御"""
    print("\n" + "="*60)
    print("测试 K-匿名性防御模块")
    print("="*60)
    
    # 创建测试图
    G = nx.karate_club_graph()
    print(f"\n原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 测试不同的k值
    k_values = [2, 3, 5]
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"测试 k = {k}")
        print(f"{'='*60}")
        
        defense = KAnonymityDefense(G, k=k)
        
        # 计算原始匿名性得分
        original_score = defense.calculate_anonymity_score(G)
        print(f"原始匿名性得分: {original_score:.2%}")
        
        # 应用K-匿名性保护
        G_protected = defense.apply_k_anonymity(method='add_edges')
        
        # 计算保护后的匿名性得分
        protected_score = defense.calculate_anonymity_score(G_protected)
        print(f"保护后匿名性得分: {protected_score:.2%}")


if __name__ == "__main__":
    test_k_anonymity()
















