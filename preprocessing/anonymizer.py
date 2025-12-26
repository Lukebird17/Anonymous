"""
匿名化模块
对社交网络图进行脱敏处理，删除所有属性，仅保留拓扑结构
"""

import networkx as nx
import numpy as np
import random
import pickle
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAnonymizer:
    """图匿名化器"""
    
    def __init__(self, edge_retention_ratio: float = 0.7,
                 add_noise_edges: bool = False,
                 noise_ratio: float = 0.05):
        """
        初始化匿名化器
        
        Args:
            edge_retention_ratio: 保留边的比例
            add_noise_edges: 是否添加噪声边
            noise_ratio: 噪声边占原图边数的比例
        """
        self.edge_retention_ratio = edge_retention_ratio
        self.add_noise_edges = add_noise_edges
        self.noise_ratio = noise_ratio
    
    def anonymize(self, G: nx.Graph) -> Tuple[nx.Graph, Dict]:
        """
        对图进行匿名化处理
        
        Args:
            G: 原始图
            
        Returns:
            (匿名图, 节点映射字典 {原始节点: 匿名节点})
        """
        logger.info("开始匿名化处理...")
        
        # 1. 创建节点ID映射（打乱顺序）
        original_nodes = list(G.nodes())
        anonymous_ids = list(range(len(original_nodes)))
        random.shuffle(anonymous_ids)
        
        node_mapping = {original_nodes[i]: anonymous_ids[i] 
                       for i in range(len(original_nodes))}
        
        # 2. 创建匿名图（删除所有节点属性）
        G_anon = nx.DiGraph() if G.is_directed() else nx.Graph()
        G_anon.add_nodes_from(anonymous_ids)
        
        # 3. 添加边（随机删除一部分）
        edges = list(G.edges())
        random.shuffle(edges)
        n_edges_to_keep = int(len(edges) * self.edge_retention_ratio)
        kept_edges = edges[:n_edges_to_keep]
        
        # 映射到匿名ID
        for u, v in kept_edges:
            u_anon = node_mapping[u]
            v_anon = node_mapping[v]
            G_anon.add_edge(u_anon, v_anon)
        
        logger.info(f"保留了 {G_anon.number_of_edges()} / {G.number_of_edges()} 条边 "
                   f"({self.edge_retention_ratio*100:.1f}%)")
        
        # 4. 可选：添加噪声边
        if self.add_noise_edges:
            self._add_noise_edges(G_anon, n_noise=int(len(edges) * self.noise_ratio))
        
        return G_anon, node_mapping
    
    def _add_noise_edges(self, G: nx.Graph, n_noise: int):
        """
        添加随机噪声边
        
        Args:
            G: 图
            n_noise: 噪声边数量
        """
        nodes = list(G.nodes())
        added = 0
        attempts = 0
        max_attempts = n_noise * 10
        
        while added < n_noise and attempts < max_attempts:
            u = random.choice(nodes)
            v = random.choice(nodes)
            
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            
            attempts += 1
        
        logger.info(f"添加了 {added} 条噪声边")
    
    def k_anonymity(self, G: nx.Graph, k: int = 5) -> nx.Graph:
        """
        实现k-匿名度（简化版）
        确保每个节点至少与k个其他节点具有相同的度
        
        Args:
            G: 图
            k: 匿名度
            
        Returns:
            k-匿名化后的图
        """
        logger.info(f"应用{k}-匿名化...")
        
        G_k = G.copy()
        
        # 计算度分布
        degrees = dict(G_k.degree())
        degree_counts = {}
        for node, degree in degrees.items():
            if degree not in degree_counts:
                degree_counts[degree] = []
            degree_counts[degree].append(node)
        
        # 对度数小于k的节点进行处理
        nodes = list(G_k.nodes())
        for degree, node_list in degree_counts.items():
            if len(node_list) < k:
                # 随机添加边使这些节点的度数相同
                for node in node_list:
                    target_degree = degree + 1
                    current_degree = G_k.degree(node)
                    
                    while current_degree < target_degree:
                        # 随机选择一个节点连接
                        other = random.choice(nodes)
                        if other != node and not G_k.has_edge(node, other):
                            G_k.add_edge(node, other)
                            current_degree += 1
        
        logger.info(f"k-匿名化完成")
        return G_k
    
    def create_ground_truth(self, original_graph: nx.Graph,
                           anonymous_graph: nx.Graph,
                           node_mapping: Dict) -> Dict:
        """
        创建地面真值（用于评估）
        
        Args:
            original_graph: 原始图
            anonymous_graph: 匿名图
            node_mapping: 节点映射
            
        Returns:
            评估信息字典
        """
        # 反向映射
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        # 提取原始图的节点列表（按原始ID排序）
        original_nodes = sorted(list(original_graph.nodes()), 
                               key=lambda x: node_mapping[x])
        
        # 提取匿名图的节点列表（已经是整数ID）
        anonymous_nodes = sorted(list(anonymous_graph.nodes()))
        
        return {
            'node_mapping': node_mapping,
            'reverse_mapping': reverse_mapping,
            'original_nodes': original_nodes,
            'anonymous_nodes': anonymous_nodes,
            'original_node_count': len(original_nodes),
            'anonymous_node_count': len(anonymous_nodes),
            'original_edge_count': original_graph.number_of_edges(),
            'anonymous_edge_count': anonymous_graph.number_of_edges(),
            'edge_retention_ratio': anonymous_graph.number_of_edges() / original_graph.number_of_edges()
        }
    
    def save_anonymized_data(self, anonymous_graph: nx.Graph,
                            ground_truth: Dict,
                            output_dir: Path):
        """
        保存匿名化数据
        
        Args:
            anonymous_graph: 匿名图
            ground_truth: 地面真值
            output_dir: 输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存匿名图
        anon_graph_path = output_dir / "anonymous_graph.gpickle"
        with open(anon_graph_path, 'wb') as f:
            pickle.dump(anonymous_graph, f)
        logger.info(f"匿名图已保存: {anon_graph_path}")
        
        # 保存地面真值
        ground_truth_path = output_dir / "ground_truth.pkl"
        with open(ground_truth_path, 'wb') as f:
            pickle.dump(ground_truth, f)
        logger.info(f"地面真值已保存: {ground_truth_path}")
        
        # 保存边列表
        edgelist_path = output_dir / "anonymous_graph.edgelist"
        nx.write_edgelist(anonymous_graph, edgelist_path)
    
    def load_anonymized_data(self, input_dir: Path) -> Tuple[nx.Graph, Dict]:
        """
        加载匿名化数据
        
        Args:
            input_dir: 输入目录
            
        Returns:
            (匿名图, 地面真值)
        """
        # 加载匿名图
        anon_graph_path = input_dir / "anonymous_graph.gpickle"
        with open(anon_graph_path, 'rb') as f:
            anonymous_graph = pickle.load(f)
        
        # 加载地面真值
        ground_truth_path = input_dir / "ground_truth.pkl"
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)
        
        logger.info(f"匿名化数据已加载: {anonymous_graph.number_of_nodes()} 节点, "
                   f"{anonymous_graph.number_of_edges()} 边")
        
        return anonymous_graph, ground_truth


def main():
    """主函数示例"""
    from preprocessing.graph_builder import GraphBuilder
    
    # 加载原始图
    builder = GraphBuilder()
    input_path = Path("data/processed/github_graph.gpickle")
    
    if not input_path.exists():
        logger.error(f"找不到文件: {input_path}")
        logger.info("请先运行 graph_builder.py 构建图")
        return
    
    G = builder.load_graph(input_path)
    
    # 匿名化
    anonymizer = GraphAnonymizer(
        edge_retention_ratio=0.7,
        add_noise_edges=True,
        noise_ratio=0.05
    )
    
    G_anon, node_mapping = anonymizer.anonymize(G)
    
    # 创建地面真值
    ground_truth = anonymizer.create_ground_truth(G, G_anon, node_mapping)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("匿名化统计")
    print("="*50)
    print(f"原始图: {ground_truth['original_node_count']} 节点, "
          f"{ground_truth['original_edge_count']} 边")
    print(f"匿名图: {ground_truth['anonymous_node_count']} 节点, "
          f"{ground_truth['anonymous_edge_count']} 边")
    print(f"边保留率: {ground_truth['edge_retention_ratio']:.2%}")
    print("="*50 + "\n")
    
    # 保存
    output_dir = Path("data/anonymized")
    anonymizer.save_anonymized_data(G_anon, ground_truth, output_dir)


if __name__ == "__main__":
    main()


