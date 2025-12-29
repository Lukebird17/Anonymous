"""
基于GraphSAGE的属性推断攻击
使用图神经网络进行节点分类/属性预测
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

# 导入GraphSAGE模型
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.graphsage import GraphSAGETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphSAGEAttributeInferenceAttack:
    """基于GraphSAGE的属性推断攻击"""
    
    def __init__(self, G: nx.Graph, node_attributes: Dict):
        """
        初始化攻击器
        
        Args:
            G: NetworkX图
            node_attributes: 节点属性字典
        """
        self.G = G
        self.node_attributes = node_attributes
        
    def extract_features_and_labels(self) -> Tuple[Dict, Dict, int, int]:
        """
        从节点属性中提取特征和标签
        
        Returns:
            (node_features, node_labels, input_dim, num_classes)
        """
        node_features = {}
        node_labels = {}
        label_to_id = {}
        next_label_id = 0
        
        for node in self.G.nodes():
            if node not in self.node_attributes:
                continue
            
            attr = self.node_attributes[node]
            
            # 提取特征
            if isinstance(attr, dict):
                # 尝试提取不同类型的特征
                if 'features' in attr:
                    # Facebook Ego网络：有特征向量
                    node_features[node] = np.array(attr['features'], dtype=np.float32)
                elif 'feature' in attr:
                    # Cora/Citeseer：有特征向量
                    node_features[node] = np.array(attr['feature'], dtype=np.float32)
                else:
                    # 使用结构特征
                    node_features[node] = self._extract_structural_features(node)
                
                # 提取标签
                if 'label' in attr:
                    label = attr['label']
                    if label not in label_to_id:
                        label_to_id[label] = next_label_id
                        next_label_id += 1
                    node_labels[node] = label_to_id[label]
                elif 'circles' in attr and attr['circles']:
                    # Facebook Ego网络：使用第一个社交圈作为标签
                    label = attr['circles'][0]
                    if label not in label_to_id:
                        label_to_id[label] = next_label_id
                        next_label_id += 1
                    node_labels[node] = label_to_id[label]
        
        # 确定特征维度
        if len(node_features) > 0:
            first_node = next(iter(node_features))
            input_dim = len(node_features[first_node])
        else:
            input_dim = 10  # 默认使用结构特征维度
        
        num_classes = len(label_to_id)
        
        logger.info(f"提取特征完成: {len(node_features)}个节点, 特征维度{input_dim}, {num_classes}个类别")
        
        return node_features, node_labels, input_dim, num_classes
    
    def _extract_structural_features(self, node: int) -> np.ndarray:
        """
        提取节点的结构特征（当没有原始特征时使用）
        
        Args:
            node: 节点ID
        
        Returns:
            结构特征向量
        """
        features = []
        
        # 度
        features.append(self.G.degree(node))
        
        # 聚类系数
        features.append(nx.clustering(self.G, node))
        
        # 邻居平均度
        neighbors = list(self.G.neighbors(node))
        if neighbors:
            neighbor_degrees = [self.G.degree(n) for n in neighbors]
            features.append(np.mean(neighbor_degrees))
            features.append(np.max(neighbor_degrees))
            features.append(np.min(neighbor_degrees))
        else:
            features.extend([0, 0, 0])
        
        # 三角形数量
        try:
            triangles = sum(nx.triangles(self.G, [node]).values())
            features.append(triangles)
        except:
            features.append(0)
        
        # 填充到10维
        while len(features) < 10:
            features.append(0)
        
        return np.array(features[:10], dtype=np.float32)
    
    def run_attack(self,
                   train_ratio: float = 0.3,
                   epochs: int = 100,
                   batch_size: int = 64,
                   hidden_dim: int = 64,
                   embed_dim: int = 32,
                   learning_rate: float = 0.01,
                   device: str = 'cpu') -> Dict:
        """
        运行GraphSAGE属性推断攻击
        
        Args:
            train_ratio: 训练集比例
            epochs: 训练轮数
            batch_size: batch大小
            hidden_dim: 隐藏层维度
            embed_dim: 嵌入维度
            learning_rate: 学习率
            device: 设备 ('cpu' or 'cuda')
        
        Returns:
            包含评估指标的字典
        """
        logger.info("="*70)
        logger.info("运行GraphSAGE属性推断攻击")
        logger.info("="*70)
        
        # 提取特征和标签
        node_features, node_labels, input_dim, num_classes = self.extract_features_and_labels()
        
        if len(node_labels) == 0:
            logger.warning("没有找到节点标签，无法进行属性推断")
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'message': '没有标签数据'
            }
        
        # 划分训练集和测试集
        labeled_nodes = list(node_labels.keys())
        np.random.shuffle(labeled_nodes)
        
        n_train = int(len(labeled_nodes) * train_ratio)
        train_nodes = labeled_nodes[:n_train]
        test_nodes = labeled_nodes[n_train:]
        
        logger.info(f"训练集: {len(train_nodes)}个节点")
        logger.info(f"测试集: {len(test_nodes)}个节点")
        
        # 创建并训练模型
        try:
            trainer = GraphSAGETrainer(
                G=self.G,
                node_features=node_features,
                node_labels=node_labels,
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                learning_rate=learning_rate,
                device=device
            )
            
            # 训练
            logger.info(f"开始训练 (epochs={epochs}, batch_size={batch_size})...")
            trainer.train(
                train_nodes=train_nodes,
                val_nodes=test_nodes[:len(test_nodes)//2] if len(test_nodes) > 1 else None,
                epochs=epochs,
                batch_size=batch_size,
                verbose=True
            )
            
            # 评估
            logger.info("评估模型...")
            test_acc = trainer.evaluate(test_nodes, batch_size=batch_size)
            
            # 获取详细预测结果
            y_pred = trainer.predict(test_nodes, batch_size=batch_size)
            y_true = np.array([node_labels[node] for node in test_nodes])
            
            # 计算指标
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            
            logger.info(f"\n评估结果:")
            logger.info(f"  - 准确率: {accuracy:.4f}")
            logger.info(f"  - F1 (macro): {f1_macro:.4f}")
            logger.info(f"  - F1 (micro): {f1_micro:.4f}")
            
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'train_nodes': len(train_nodes),
                'test_nodes': len(test_nodes),
                'num_classes': num_classes,
                'model': trainer  # 返回训练好的模型
            }
            
        except Exception as e:
            logger.error(f"GraphSAGE训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'message': f'训练失败: {str(e)}'
            }


def test_graphsage_on_cora():
    """在Cora数据集上测试GraphSAGE"""
    from data.dataset_loader import DatasetLoader
    
    logger.info("在Cora数据集上测试GraphSAGE...")
    
    # 加载数据
    loader = DatasetLoader()
    G, attributes = loader.load_cora()
    
    if G is None:
        logger.error("加载Cora数据集失败")
        return
    
    # 创建攻击器
    attacker = GraphSAGEAttributeInferenceAttack(G, attributes)
    
    # 运行攻击
    results = attacker.run_attack(
        train_ratio=0.3,
        epochs=50,
        batch_size=64,
        hidden_dim=64,
        embed_dim=32,
        learning_rate=0.01,
        device='cpu'
    )
    
    logger.info(f"\n最终结果:")
    logger.info(f"  准确率: {results['accuracy']:.2%}")
    logger.info(f"  F1 (macro): {results['f1_macro']:.4f}")


if __name__ == "__main__":
    test_graphsage_on_cora()

