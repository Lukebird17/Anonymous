"""
GraphSAGE (Graph Sample and Aggregate) 实现
用于属性推断攻击

参考论文：
Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeanAggregator(nn.Module):
    """均值聚合器 - GraphSAGE的核心组件"""
    
    def __init__(self, in_features: int, out_features: int):
        super(MeanAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 自身特征变换
        self.linear_self = nn.Linear(in_features, out_features)
        # 邻居特征变换
        self.linear_neighbor = nn.Linear(in_features, out_features)
        
    def forward(self, self_features: torch.Tensor, neighbor_features: torch.Tensor):
        """
        前向传播
        
        Args:
            self_features: 节点自身特征 [batch_size, in_features]
            neighbor_features: 邻居特征 [batch_size, num_neighbors, in_features]
        
        Returns:
            聚合后的特征 [batch_size, out_features]
        """
        # 聚合邻居特征（均值）
        neighbor_agg = torch.mean(neighbor_features, dim=1)  # [batch_size, in_features]
        
        # 分别变换自身和邻居特征
        self_transformed = self.linear_self(self_features)
        neighbor_transformed = self.linear_neighbor(neighbor_agg)
        
        # 拼接并激活
        combined = self_transformed + neighbor_transformed
        
        return combined


class GraphSAGE(nn.Module):
    """GraphSAGE模型 - 两层GNN"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 16,
                 dropout: float = 0.5):
        """
        初始化GraphSAGE模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            dropout: Dropout率
        """
        super(GraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 第一层聚合器（处理原始特征）
        self.agg1 = MeanAggregator(input_dim, hidden_dim)
        # 第二层聚合器（第一层输出 + 原始邻居特征）
        # 注意：这里简化了实现，直接使用原始特征维度
        self.agg2 = MeanAggregator(hidden_dim, output_dim)
        # 为一跳邻居添加单独的变换层
        self.neighbor_transform = nn.Linear(input_dim, hidden_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, 
                node_features: torch.Tensor,
                neighbor_features_1hop: torch.Tensor,
                neighbor_features_2hop: torch.Tensor):
        """
        前向传播（两层聚合）
        
        Args:
            node_features: 节点特征 [batch_size, input_dim]
            neighbor_features_1hop: 一跳邻居特征 [batch_size, num_neighbors_1, input_dim]
            neighbor_features_2hop: 二跳邻居特征 [batch_size, num_neighbors_2, input_dim]
        
        Returns:
            节点嵌入 [batch_size, output_dim]
        """
        # 第一层聚合（使用二跳邻居）
        h1 = self.agg1(node_features, neighbor_features_2hop)
        h1 = F.relu(h1)
        h1 = self.dropout_layer(h1)
        
        # 第二层聚合（使用一跳邻居）
        # 先将一跳邻居特征变换到hidden_dim维度
        batch_size, num_neighbors, input_dim = neighbor_features_1hop.shape
        neighbor_1hop_flat = neighbor_features_1hop.view(-1, input_dim)  # [batch*neighbors, input_dim]
        neighbor_1hop_transformed = self.neighbor_transform(neighbor_1hop_flat)  # [batch*neighbors, hidden_dim]
        neighbor_1hop_transformed = neighbor_1hop_transformed.view(batch_size, num_neighbors, self.hidden_dim)  # [batch, neighbors, hidden_dim]
        
        h2 = self.agg2(h1, neighbor_1hop_transformed)
        
        # L2归一化
        h2 = F.normalize(h2, p=2, dim=1)
        
        return h2


class GraphSAGEClassifier(nn.Module):
    """基于GraphSAGE的节点分类器"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 embed_dim: int = 32,
                 num_classes: int = 7,
                 dropout: float = 0.5):
        """
        初始化分类器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            embed_dim: 嵌入维度
            num_classes: 类别数
            dropout: Dropout率
        """
        super(GraphSAGEClassifier, self).__init__()
        
        # GraphSAGE嵌入层
        self.graphsage = GraphSAGE(input_dim, hidden_dim, embed_dim, dropout)
        
        # 分类层
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, node_features, neighbor_features_1hop, neighbor_features_2hop):
        """前向传播"""
        # 获取节点嵌入
        embeddings = self.graphsage(node_features, neighbor_features_1hop, neighbor_features_2hop)
        
        # 分类
        logits = self.classifier(embeddings)
        
        return logits, embeddings


class GraphSAGETrainer:
    """GraphSAGE训练器"""
    
    def __init__(self,
                 G: nx.Graph,
                 node_features: Dict,
                 node_labels: Dict,
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 64,
                 embed_dim: int = 32,
                 num_neighbors: int = 10,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            G: NetworkX图
            node_features: 节点特征字典 {node_id: feature_vector}
            node_labels: 节点标签字典 {node_id: label}
            input_dim: 输入特征维度
            num_classes: 类别数
            hidden_dim: 隐藏层维度
            embed_dim: 嵌入维度
            num_neighbors: 采样邻居数
            learning_rate: 学习率
            device: 设备 ('cpu' or 'cuda')
        """
        self.G = G
        self.node_features = node_features
        self.node_labels = node_labels
        self.num_neighbors = num_neighbors
        self.device = torch.device(device)
        
        # 创建模型
        self.model = GraphSAGEClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=0.5
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"GraphSAGE模型已创建 (输入维度: {input_dim}, 类别数: {num_classes})")
        logger.info(f"设备: {self.device}")
        
    def sample_neighbors(self, node: int, k: int = None) -> List[int]:
        """
        采样k个邻居（均匀采样）
        
        Args:
            node: 节点ID
            k: 采样数量，默认使用self.num_neighbors
        
        Returns:
            邻居列表
        """
        if k is None:
            k = self.num_neighbors
        
        neighbors = list(self.G.neighbors(node))
        
        if len(neighbors) == 0:
            # 如果没有邻居，返回自己
            return [node] * k
        elif len(neighbors) >= k:
            # 随机采样k个
            return np.random.choice(neighbors, k, replace=False).tolist()
        else:
            # 不足k个，重复采样
            return np.random.choice(neighbors, k, replace=True).tolist()
    
    def prepare_batch(self, nodes: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        准备一个batch的数据
        
        Args:
            nodes: 节点列表
        
        Returns:
            (node_features, neighbor_features_1hop, neighbor_features_2hop, labels)
        """
        batch_size = len(nodes)
        input_dim = len(self.node_features[nodes[0]])
        
        # 初始化张量
        node_feat = np.zeros((batch_size, input_dim))
        neighbor_feat_1hop = np.zeros((batch_size, self.num_neighbors, input_dim))
        neighbor_feat_2hop = np.zeros((batch_size, self.num_neighbors, input_dim))
        labels = np.zeros(batch_size, dtype=np.int64)
        
        for i, node in enumerate(nodes):
            # 节点自身特征
            node_feat[i] = self.node_features[node]
            
            # 标签
            if node in self.node_labels:
                labels[i] = self.node_labels[node]
            
            # 采样一跳邻居
            neighbors_1hop = self.sample_neighbors(node)
            for j, neighbor in enumerate(neighbors_1hop):
                if neighbor in self.node_features:
                    neighbor_feat_1hop[i, j] = self.node_features[neighbor]
            
            # 采样二跳邻居（从一跳邻居的邻居中采样）
            neighbors_2hop_all = []
            for neighbor in neighbors_1hop:
                neighbors_2hop_all.extend(self.sample_neighbors(neighbor, k=2))
            
            if len(neighbors_2hop_all) >= self.num_neighbors:
                neighbors_2hop = np.random.choice(neighbors_2hop_all, self.num_neighbors, replace=False)
            else:
                neighbors_2hop = np.random.choice(neighbors_2hop_all, self.num_neighbors, replace=True)
            
            for j, neighbor in enumerate(neighbors_2hop):
                if neighbor in self.node_features:
                    neighbor_feat_2hop[i, j] = self.node_features[neighbor]
        
        # 转换为PyTorch张量
        node_feat = torch.FloatTensor(node_feat).to(self.device)
        neighbor_feat_1hop = torch.FloatTensor(neighbor_feat_1hop).to(self.device)
        neighbor_feat_2hop = torch.FloatTensor(neighbor_feat_2hop).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        return node_feat, neighbor_feat_1hop, neighbor_feat_2hop, labels
    
    def train(self, 
              train_nodes: List[int],
              val_nodes: List[int] = None,
              epochs: int = 100,
              batch_size: int = 64,
              verbose: bool = True):
        """
        训练模型
        
        Args:
            train_nodes: 训练节点列表
            val_nodes: 验证节点列表
            epochs: 训练轮数
            batch_size: batch大小
            verbose: 是否打印训练信息
        """
        self.model.train()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 随机打乱训练节点
            np.random.shuffle(train_nodes)
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Mini-batch训练
            for i in range(0, len(train_nodes), batch_size):
                batch_nodes = train_nodes[i:i+batch_size]
                
                # 准备数据
                node_feat, neighbor_feat_1hop, neighbor_feat_2hop, labels = self.prepare_batch(batch_nodes)
                
                # 前向传播
                self.optimizer.zero_grad()
                logits, _ = self.model(node_feat, neighbor_feat_1hop, neighbor_feat_2hop)
                
                # 计算损失
                loss = self.criterion(logits, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            train_acc = correct / total if total > 0 else 0
            
            # 验证
            val_acc = 0.0
            if val_nodes is not None and len(val_nodes) > 0:
                val_acc = self.evaluate(val_nodes, batch_size)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # 打印信息
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_nodes):.4f}, "
                           f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, test_nodes: List[int], batch_size: int = 64) -> float:
        """
        评估模型
        
        Args:
            test_nodes: 测试节点列表
            batch_size: batch大小
        
        Returns:
            准确率
        """
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_nodes), batch_size):
                batch_nodes = test_nodes[i:i+batch_size]
                
                # 准备数据
                node_feat, neighbor_feat_1hop, neighbor_feat_2hop, labels = self.prepare_batch(batch_nodes)
                
                # 前向传播
                logits, _ = self.model(node_feat, neighbor_feat_1hop, neighbor_feat_2hop)
                
                # 预测
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def predict(self, test_nodes: List[int], batch_size: int = 64) -> np.ndarray:
        """
        预测节点标签
        
        Args:
            test_nodes: 测试节点列表
            batch_size: batch大小
        
        Returns:
            预测标签数组
        """
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(test_nodes), batch_size):
                batch_nodes = test_nodes[i:i+batch_size]
                
                # 准备数据
                node_feat, neighbor_feat_1hop, neighbor_feat_2hop, _ = self.prepare_batch(batch_nodes)
                
                # 前向传播
                logits, _ = self.model(node_feat, neighbor_feat_1hop, neighbor_feat_2hop)
                
                # 预测
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def get_embeddings(self, nodes: List[int], batch_size: int = 64) -> np.ndarray:
        """
        获取节点嵌入
        
        Args:
            nodes: 节点列表
            batch_size: batch大小
        
        Returns:
            节点嵌入矩阵 [num_nodes, embed_dim]
        """
        self.model.eval()
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(nodes), batch_size):
                batch_nodes = nodes[i:i+batch_size]
                
                # 准备数据
                node_feat, neighbor_feat_1hop, neighbor_feat_2hop, _ = self.prepare_batch(batch_nodes)
                
                # 前向传播
                _, embeds = self.model(node_feat, neighbor_feat_1hop, neighbor_feat_2hop)
                
                embeddings.append(embeds.cpu().numpy())
        
        return np.vstack(embeddings)

