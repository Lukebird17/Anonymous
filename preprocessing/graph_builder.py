"""
图构建模块
将爬取的原始数据转换为NetworkX图结构
"""

import json
import networkx as nx
import pickle
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """图构建器"""
    
    def __init__(self):
        self.graph = None
    
    def build_from_weibo(self, data_path: Path) -> nx.DiGraph:
        """
        从微博数据构建有向图
        
        Args:
            data_path: 微博数据文件路径
            
        Returns:
            NetworkX有向图
        """
        logger.info(f"从微博数据构建图: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点及其属性
        for uid, user_info in data['users'].items():
            G.add_node(uid, **user_info)
        
        # 添加边（关注关系）
        for from_uid, to_uid in data['edges']:
            if from_uid in G.nodes and to_uid in G.nodes:
                G.add_edge(from_uid, to_uid)
        
        logger.info(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        self.graph = G
        return G
    
    def build_from_github(self, data_path: Path, 
                         use_starred_repos: bool = False) -> nx.DiGraph:
        """
        从GitHub数据构建图
        
        Args:
            data_path: GitHub数据文件路径
            use_starred_repos: 是否使用Star的仓库构建异构图
            
        Returns:
            NetworkX有向图
        """
        logger.info(f"从GitHub数据构建图: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点及其属性
        for username, user_info in data['users'].items():
            G.add_node(username, node_type='user', **user_info)
        
        # 添加边（关注关系）
        for edge in data['edges']:
            if len(edge) == 3:
                from_user, to_user, edge_type = edge
            else:
                from_user, to_user = edge
                edge_type = 'follow'
            
            if from_user in G.nodes and to_user in G.nodes:
                G.add_edge(from_user, to_user, edge_type=edge_type)
        
        # 可选：添加Star关系构建异构图
        if use_starred_repos and 'starred_repos' in data:
            for username, repos in data['starred_repos'].items():
                for repo in repos:
                    # 添加仓库节点
                    if repo not in G.nodes:
                        G.add_node(repo, node_type='repo')
                    
                    # 添加Star边
                    G.add_edge(username, repo, edge_type='star')
        
        logger.info(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        self.graph = G
        return G
    
    def compute_node_features(self, G: nx.Graph = None) -> nx.Graph:
        """
        计算节点的拓扑特征
        
        Args:
            G: NetworkX图，如果为None则使用self.graph
            
        Returns:
            添加了特征的图
        """
        if G is None:
            G = self.graph
        
        logger.info("计算节点拓扑特征...")
        
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        
        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # 接近中心性
        closeness_centrality = nx.closeness_centrality(G)
        
        # PageRank
        pagerank = nx.pagerank(G)
        
        # 聚集系数
        if G.is_directed():
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
        else:
            clustering = nx.clustering(G)
        
        # 将特征添加到节点
        for node in G.nodes():
            G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
            G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
            G.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
            G.nodes[node]['pagerank'] = pagerank.get(node, 0)
            G.nodes[node]['clustering'] = clustering.get(node, 0)
            G.nodes[node]['in_degree'] = G.in_degree(node) if G.is_directed() else G.degree(node)
            G.nodes[node]['out_degree'] = G.out_degree(node) if G.is_directed() else G.degree(node)
        
        logger.info("特征计算完成")
        return G
    
    def extract_largest_component(self, G: nx.Graph = None) -> nx.Graph:
        """
        提取最大连通分量
        
        Args:
            G: NetworkX图
            
        Returns:
            最大连通分量子图
        """
        if G is None:
            G = self.graph
        
        if G.is_directed():
            # 有向图：提取最大弱连通分量
            components = list(nx.weakly_connected_components(G))
        else:
            components = list(nx.connected_components(G))
        
        largest_component = max(components, key=len)
        subgraph = G.subgraph(largest_component).copy()
        
        logger.info(f"最大连通分量: {subgraph.number_of_nodes()} 节点, {subgraph.number_of_edges()} 边")
        
        return subgraph
    
    def save_graph(self, G: nx.Graph, output_path: Path):
        """
        保存图到文件
        
        Args:
            G: NetworkX图
            output_path: 输出路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为pickle格式
        with open(output_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"图已保存到: {output_path}")
        
        # 同时保存为edge list格式（用于其他工具）
        edgelist_path = output_path.with_suffix('.edgelist')
        nx.write_edgelist(G, edgelist_path)
        logger.info(f"边列表已保存到: {edgelist_path}")
    
    def load_graph(self, input_path: Path) -> nx.Graph:
        """
        从文件加载图
        
        Args:
            input_path: 输入路径
            
        Returns:
            NetworkX图
        """
        with open(input_path, 'rb') as f:
            G = pickle.load(f)
        
        logger.info(f"图已加载: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        self.graph = G
        return G
    
    def print_graph_stats(self, G: nx.Graph = None):
        """
        打印图的统计信息
        
        Args:
            G: NetworkX图
        """
        if G is None:
            G = self.graph
        
        print("\n" + "="*50)
        print("图统计信息")
        print("="*50)
        print(f"节点数: {G.number_of_nodes()}")
        print(f"边数: {G.number_of_edges()}")
        print(f"图类型: {'有向图' if G.is_directed() else '无向图'}")
        
        if G.is_directed():
            print(f"平均入度: {sum(d for _, d in G.in_degree()) / G.number_of_nodes():.2f}")
            print(f"平均出度: {sum(d for _, d in G.out_degree()) / G.number_of_nodes():.2f}")
        else:
            print(f"平均度: {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}")
        
        # 密度
        density = nx.density(G)
        print(f"密度: {density:.6f}")
        
        # 连通分量
        if G.is_directed():
            n_components = nx.number_weakly_connected_components(G)
            print(f"弱连通分量数: {n_components}")
        else:
            n_components = nx.number_connected_components(G)
            print(f"连通分量数: {n_components}")
        
        print("="*50 + "\n")


def main():
    """主函数示例"""
    builder = GraphBuilder()
    
    # 示例1: 从微博数据构建图
    # weibo_data = Path("data/raw/weibo_data.json")
    # if weibo_data.exists():
    #     G = builder.build_from_weibo(weibo_data)
    #     G = builder.compute_node_features(G)
    #     G = builder.extract_largest_component(G)
    #     builder.print_graph_stats(G)
    #     builder.save_graph(G, Path("data/processed/weibo_graph.gpickle"))
    
    # 示例2: 从GitHub数据构建图
    github_data = Path("data/raw/github_data.json")
    if github_data.exists():
        G = builder.build_from_github(github_data, use_starred_repos=False)
        G = builder.compute_node_features(G)
        G = builder.extract_largest_component(G)
        builder.print_graph_stats(G)
        builder.save_graph(G, Path("data/processed/github_graph.gpickle"))


if __name__ == "__main__":
    main()


