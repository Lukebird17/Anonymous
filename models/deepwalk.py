"""
DeepWalk图嵌入实现
使用随机游走 + Skip-gram模型学习节点的向量表示
"""

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from typing import List
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepWalk:
    """DeepWalk图嵌入模型"""
    
    def __init__(self, 
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 window_size: int = 10,
                 workers: int = 4,
                 epochs: int = 5):
        """
        初始化DeepWalk模型
        
        Args:
            dimensions: 嵌入向量维度
            walk_length: 每次游走的长度
            num_walks: 每个节点的游走次数
            window_size: Skip-gram窗口大小
            workers: 并行worker数量
            epochs: 训练轮数
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.epochs = epochs
        self.model = None
    
    def _random_walk(self, graph: nx.Graph, start_node) -> List:
        """
        从指定节点开始进行随机游走
        
        Args:
            graph: NetworkX图
            start_node: 起始节点
            
        Returns:
            游走路径（节点列表）
        """
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(graph.neighbors(current))
            
            if len(neighbors) == 0:
                break
            
            next_node = random.choice(neighbors)
            walk.append(next_node)
        
        return walk
    
    def generate_walks(self, graph: nx.Graph) -> List[List]:
        """
        生成所有随机游走序列
        
        Args:
            graph: NetworkX图
            
        Returns:
            游走序列列表
        """
        logger.info(f"生成随机游走序列 (每个节点 {self.num_walks} 次)...")
        
        walks = []
        nodes = list(graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(graph, node)
                # 将节点转换为字符串（Word2Vec要求）
                walks.append([str(n) for n in walk])
        
        logger.info(f"生成了 {len(walks)} 条游走序列")
        return walks
    
    def fit(self, graph: nx.Graph):
        """
        训练DeepWalk模型
        
        Args:
            graph: NetworkX图
        """
        logger.info("训练DeepWalk模型...")
        
        # 生成随机游走
        walks = self.generate_walks(graph)
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=self.epochs
        )
        
        logger.info("DeepWalk模型训练完成")
    
    def train(self, graph: nx.Graph) -> np.ndarray:
        """
        训练DeepWalk模型并返回嵌入（兼容接口）
        
        Args:
            graph: NetworkX图
            
        Returns:
            嵌入矩阵
        """
        self.fit(graph)
        return self.get_embeddings()
    
    def get_embedding(self, node) -> np.ndarray:
        """
        获取单个节点的嵌入向量
        
        Args:
            node: 节点ID
            
        Returns:
            嵌入向量
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        try:
            return self.model.wv[str(node)]
        except KeyError:
            # 节点不存在，返回零向量
            return np.zeros(self.dimensions)
    
    def get_embeddings(self, nodes: List = None) -> np.ndarray:
        """
        获取多个节点的嵌入矩阵
        
        Args:
            nodes: 节点列表，如果为None则返回所有节点
            
        Returns:
            嵌入矩阵 [n_nodes, dimensions]
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit() 方法")
        
        if nodes is None:
            nodes = list(self.model.wv.index_to_key)
            nodes = [int(n) if n.isdigit() else n for n in nodes]
        
        embeddings = np.array([self.get_embedding(node) for node in nodes])
        return embeddings
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练")
        
        self.model.save(path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        self.model = Word2Vec.load(path)
        logger.info(f"模型已加载: {path}")


def main():
    """主函数示例"""
    import pickle
    from pathlib import Path
    
    # 加载图
    graph_path = Path("data/processed/github_graph.gpickle")
    if not graph_path.exists():
        logger.error(f"找不到文件: {graph_path}")
        return
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    # 转换为无向图（DeepWalk通常用于无向图）
    if G.is_directed():
        G = G.to_undirected()
    
    logger.info(f"图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 训练DeepWalk
    model = DeepWalk(
        dimensions=128,
        walk_length=80,
        num_walks=10,
        window_size=10,
        workers=4,
        epochs=5
    )
    
    model.fit(G)
    
    # 获取所有节点的嵌入
    nodes = list(G.nodes())
    embeddings = model.get_embeddings(nodes)
    
    logger.info(f"嵌入矩阵形状: {embeddings.shape}")
    
    # 保存模型
    model_path = Path("models") / "deepwalk_model.bin"
    model_path.parent.mkdir(exist_ok=True)
    model.save_model(str(model_path))
    
    # 保存嵌入
    embeddings_path = Path("models") / "deepwalk_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"嵌入已保存到: {embeddings_path}")


if __name__ == "__main__":
    main()


# 别名，用于兼容性
DeepWalkModel = DeepWalk

