"""
可视化模块
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """图可视化器"""
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
    
    def plot_graph(self, G: nx.Graph, output_path: Path = None,
                   title: str = "Social Network Graph",
                   node_color: str = 'lightblue',
                   layout: str = 'spring'):
        """
        绘制图
        
        Args:
            G: NetworkX图
            output_path: 保存路径
            title: 标题
            node_color: 节点颜色
            layout: 布局算法
        """
        plt.figure(figsize=self.figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # 绘制
        nx.draw_networkx_nodes(G, pos, node_color=node_color, 
                              node_size=100, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_degree_distribution(self, G: nx.Graph, output_path: Path = None):
        """绘制度分布"""
        degrees = [d for _, d in G.degree()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('度 (Degree)', fontsize=12)
        plt.ylabel('节点数量 (Number of Nodes)', fontsize=12)
        plt.title('度分布 (Degree Distribution)', fontsize=14)
        plt.grid(alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"度分布图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, G_orig: nx.Graph, G_anon: nx.Graph,
                       output_path: Path = None):
        """对比原始图和匿名图"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 原始图
        pos1 = nx.spring_layout(G_orig, k=0.5, iterations=30)
        axes[0].set_title('原始图 (Original Graph)', fontsize=14)
        nx.draw_networkx_nodes(G_orig, pos1, ax=axes[0], 
                              node_color='lightblue', node_size=50, alpha=0.7)
        nx.draw_networkx_edges(G_orig, pos1, ax=axes[0], alpha=0.2, width=0.5)
        axes[0].axis('off')
        
        # 匿名图
        pos2 = nx.spring_layout(G_anon, k=0.5, iterations=30)
        axes[1].set_title('匿名图 (Anonymous Graph)', fontsize=14)
        nx.draw_networkx_nodes(G_anon, pos2, ax=axes[1],
                              node_color='lightcoral', node_size=50, alpha=0.7)
        nx.draw_networkx_edges(G_anon, pos2, ax=axes[1], alpha=0.2, width=0.5)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"对比图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_attack_results(self, results: Dict, output_path: Path = None):
        """绘制攻击结果"""
        methods = list(results.keys())
        accuracies = [results[m].get('accuracy', 0) for m in methods]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率对比
        axes[0].bar(methods, accuracies, alpha=0.7, color='steelblue')
        axes[0].set_ylabel('准确率 (Accuracy)', fontsize=12)
        axes[0].set_title('不同方法的准确率对比', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(alpha=0.3, axis='y')
        
        # Top-K准确率
        if 'top_k' in results[methods[0]]:
            k_values = sorted(results[methods[0]]['top_k'].keys())
            for method in methods:
                top_k_accs = [results[method]['top_k'][k] for k in k_values]
                axes[1].plot(k_values, top_k_accs, marker='o', label=method)
            
            axes[1].set_xlabel('K', fontsize=12)
            axes[1].set_ylabel('Top-K准确率', fontsize=12)
            axes[1].set_title('Top-K准确率对比', fontsize=14)
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"结果图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()


