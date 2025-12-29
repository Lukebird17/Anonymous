"""
属性推断攻击 - 利用同质性原理推断节点隐藏属性
基于 GraphSAGE 或简单的标签传播算法
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


class AttributeInferenceAttack:
    """属性推断攻击器"""
    
    def __init__(self, G: nx.Graph, node_attributes: Dict):
        """
        初始化属性推断攻击器
        
        Args:
            G: 图对象
            node_attributes: 节点属性字典 {node_id: {'label': 'xxx', ...}}
        """
        self.G = G
        self.node_attributes = node_attributes
        
    def extract_structural_features(self, node: int) -> np.ndarray:
        """
        提取节点的结构特征
        
        Args:
            node: 节点ID
            
        Returns:
            特征向量
        """
        features = []
        
        # 基础拓扑特征
        features.append(self.G.degree(node))  # 度
        
        # 中心性特征
        try:
            # 使用预计算的中心性（如果有）
            if not hasattr(self, '_betweenness'):
                self._betweenness = nx.betweenness_centrality(self.G)
            features.append(self._betweenness.get(node, 0))
        except:
            features.append(0)
        
        try:
            if not hasattr(self, '_closeness'):
                self._closeness = nx.closeness_centrality(self.G)
            features.append(self._closeness.get(node, 0))
        except:
            features.append(0)
        
        try:
            if not hasattr(self, '_pagerank'):
                self._pagerank = nx.pagerank(self.G)
            features.append(self._pagerank.get(node, 0))
        except:
            features.append(0)
        
        # 聚类系数
        features.append(nx.clustering(self.G, node))
        
        # 邻居特征聚合
        neighbors = list(self.G.neighbors(node))
        if neighbors:
            # 邻居平均度
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
        
        return np.array(features)
    
    def prepare_training_data(self, labeled_nodes: List[int], 
                             attribute_key: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            labeled_nodes: 已标注节点列表
            attribute_key: 属性键名
            
        Returns:
            (X, y) 特征矩阵和标签向量
        """
        X = []
        y = []
        
        for node in labeled_nodes:
            if node not in self.G:
                continue
            
            # 提取结构特征
            features = self.extract_structural_features(node)
            
            # 如果节点有原始特征，拼接上
            if node in self.node_attributes and 'features' in self.node_attributes[node]:
                original_features = self.node_attributes[node]['features']
                if isinstance(original_features, np.ndarray):
                    features = np.concatenate([features, original_features])
            
            X.append(features)
            
            # 获取标签
            if node in self.node_attributes and attribute_key in self.node_attributes[node]:
                y.append(self.node_attributes[node][attribute_key])
            else:
                continue
        
        return np.array(X), np.array(y)
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        model_type: str = 'rf'):
        """
        训练分类器
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            model_type: 模型类型 ('rf', 'lr')
            
        Returns:
            训练好的模型
        """
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == 'lr':
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def infer_attributes(self, unlabeled_nodes: List[int], model,
                        attribute_key: str = 'label') -> Dict[int, str]:
        """
        推断未标注节点的属性
        
        Args:
            unlabeled_nodes: 未标注节点列表
            model: 训练好的模型
            attribute_key: 属性键名
            
        Returns:
            {node_id: predicted_label} 字典
        """
        predictions = {}
        
        X_test = []
        valid_nodes = []
        
        for node in unlabeled_nodes:
            if node not in self.G:
                continue
            
            features = self.extract_structural_features(node)
            
            # 拼接原始特征
            if node in self.node_attributes and 'features' in self.node_attributes[node]:
                original_features = self.node_attributes[node]['features']
                if isinstance(original_features, np.ndarray):
                    features = np.concatenate([features, original_features])
            
            X_test.append(features)
            valid_nodes.append(node)
        
        if len(X_test) == 0:
            return predictions
        
        X_test = np.array(X_test)
        y_pred = model.predict(X_test)
        
        for node, pred in zip(valid_nodes, y_pred):
            predictions[node] = pred
        
        return predictions
    
    def evaluate_inference(self, test_nodes: List[int], predictions: Dict[int, str],
                          attribute_key: str = 'label') -> Dict:
        """
        评估属性推断效果
        
        Args:
            test_nodes: 测试节点列表
            predictions: 预测结果字典
            attribute_key: 属性键名
            
        Returns:
            评估指标字典
        """
        y_true = []
        y_pred = []
        
        for node in test_nodes:
            if node not in predictions:
                continue
            if node not in self.node_attributes or attribute_key not in self.node_attributes[node]:
                continue
            
            y_true.append(self.node_attributes[node][attribute_key])
            y_pred.append(predictions[node])
        
        if len(y_true) == 0:
            return {'error': '没有有效的测试样本'}
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'n_test_samples': len(y_true)
        }
        
        return metrics
    
    def run_complete_attack(self, train_ratio: float = 0.3, 
                          attribute_key: str = 'label',
                          model_type: str = 'rf') -> Dict:
        """
        运行完整的属性推断攻击
        
        Args:
            train_ratio: 训练集比例（已知标签的节点比例）
            attribute_key: 要推断的属性键名
            model_type: 模型类型
            
        Returns:
            攻击结果字典
        """
        # 1. 获取有标签的节点
        labeled_nodes = [node for node in self.G.nodes() 
                        if node in self.node_attributes 
                        and attribute_key in self.node_attributes[node]]
        
        if len(labeled_nodes) == 0:
            return {'error': '没有找到有标签的节点'}
        
        # 2. 划分训练集和测试集
        np.random.shuffle(labeled_nodes)
        n_train = int(len(labeled_nodes) * train_ratio)
        train_nodes = labeled_nodes[:n_train]
        test_nodes = labeled_nodes[n_train:]
        
        print(f"训练集: {len(train_nodes)} 节点, 测试集: {len(test_nodes)} 节点")
        
        # 3. 准备训练数据
        X_train, y_train = self.prepare_training_data(train_nodes, attribute_key)
        print(f"特征维度: {X_train.shape}")
        print(f"类别分布: {np.unique(y_train, return_counts=True)}")
        
        # 4. 训练模型
        print(f"训练 {model_type.upper()} 模型...")
        model = self.train_classifier(X_train, y_train, model_type)
        
        # 5. 推断测试集
        predictions = self.infer_attributes(test_nodes, model, attribute_key)
        
        # 6. 评估
        metrics = self.evaluate_inference(test_nodes, predictions, attribute_key)
        
        # 7. 返回完整结果
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'train_size': len(train_nodes),
            'test_size': len(test_nodes),
            'model_type': model_type
        }
        
        return results


class LabelPropagationAttack:
    """基于标签传播的属性推断（更简单的基准方法）"""
    
    def __init__(self, G: nx.Graph, node_attributes: Dict):
        """
        初始化标签传播攻击器
        
        Args:
            G: 图对象
            node_attributes: 节点属性字典
        """
        self.G = G
        self.node_attributes = node_attributes
    
    def propagate_labels(self, labeled_nodes: Dict[int, str], 
                        max_iterations: int = 10) -> Dict[int, str]:
        """
        标签传播算法
        
        Args:
            labeled_nodes: 已标注节点 {node_id: label}
            max_iterations: 最大迭代次数
            
        Returns:
            所有节点的标签预测
        """
        # 初始化标签
        labels = {node: None for node in self.G.nodes()}
        labels.update(labeled_nodes)
        
        # 迭代传播
        for iteration in range(max_iterations):
            new_labels = labels.copy()
            changed = 0
            
            for node in self.G.nodes():
                if node in labeled_nodes:  # 已知标签不变
                    continue
                
                # 统计邻居标签
                neighbor_labels = []
                for neighbor in self.G.neighbors(node):
                    if labels[neighbor] is not None:
                        neighbor_labels.append(labels[neighbor])
                
                if neighbor_labels:
                    # 多数投票
                    from collections import Counter
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    if new_labels[node] != most_common:
                        new_labels[node] = most_common
                        changed += 1
            
            labels = new_labels
            print(f"迭代 {iteration+1}: {changed} 个节点标签改变")
            
            if changed == 0:
                break
        
        return labels
    
    def run_attack(self, train_ratio: float = 0.3, 
                  attribute_key: str = 'label') -> Dict:
        """
        运行标签传播攻击
        
        Args:
            train_ratio: 已知标签的节点比例
            attribute_key: 属性键名
            
        Returns:
            攻击结果
        """
        # 获取有标签的节点
        labeled_nodes_all = {node: self.node_attributes[node][attribute_key]
                            for node in self.G.nodes() 
                            if node in self.node_attributes 
                            and attribute_key in self.node_attributes[node]}
        
        if len(labeled_nodes_all) == 0:
            return {'error': '没有找到有标签的节点'}
        
        # 划分训练/测试
        all_labeled = list(labeled_nodes_all.keys())
        np.random.shuffle(all_labeled)
        n_train = int(len(all_labeled) * train_ratio)
        train_nodes = all_labeled[:n_train]
        test_nodes = all_labeled[n_train:]
        
        labeled_nodes_train = {node: labeled_nodes_all[node] for node in train_nodes}
        
        print(f"训练集: {len(train_nodes)} 节点, 测试集: {len(test_nodes)} 节点")
        
        # 标签传播
        predictions_all = self.propagate_labels(labeled_nodes_train)
        
        # 提取测试集预测
        predictions = {node: predictions_all[node] for node in test_nodes 
                      if predictions_all[node] is not None}
        
        # 评估
        y_true = [labeled_nodes_all[node] for node in test_nodes 
                 if node in predictions]
        y_pred = [predictions[node] for node in test_nodes 
                 if node in predictions]
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'n_test_samples': len(y_true)
        }
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'train_size': len(train_nodes),
            'test_size': len(test_nodes)
        }


def test_attribute_inference():
    """测试属性推断攻击"""
    print("\n" + "="*60)
    print("测试属性推断攻击")
    print("="*60)
    
    # 创建测试数据 - 空手道俱乐部图
    G = nx.karate_club_graph()
    
    # 添加模拟标签（根据实际社团划分）
    node_attributes = {}
    for node in G.nodes():
        # 空手道俱乐部有两个派系（Mr. Hi vs Officer）
        club = G.nodes[node]['club']
        node_attributes[node] = {'label': club}
    
    print(f"\n图信息: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"类别分布: {set(attr['label'] for attr in node_attributes.values())}")
    
    # 测试1: 基于结构特征的分类器
    print("\n【测试1】基于结构特征的随机森林分类器")
    print("-" * 60)
    attacker = AttributeInferenceAttack(G, node_attributes)
    results_rf = attacker.run_complete_attack(train_ratio=0.3, model_type='rf')
    print(f"准确率: {results_rf['metrics']['accuracy']:.4f}")
    print(f"F1 (macro): {results_rf['metrics']['f1_macro']:.4f}")
    
    # 测试2: 逻辑回归
    print("\n【测试2】逻辑回归分类器")
    print("-" * 60)
    results_lr = attacker.run_complete_attack(train_ratio=0.3, model_type='lr')
    print(f"准确率: {results_lr['metrics']['accuracy']:.4f}")
    print(f"F1 (macro): {results_lr['metrics']['f1_macro']:.4f}")
    
    # 测试3: 标签传播
    print("\n【测试3】标签传播算法")
    print("-" * 60)
    lp_attacker = LabelPropagationAttack(G, node_attributes)
    results_lp = lp_attacker.run_attack(train_ratio=0.3)
    print(f"准确率: {results_lp['metrics']['accuracy']:.4f}")
    print(f"F1 (macro): {results_lp['metrics']['f1_macro']:.4f}")
    
    # 测试不同训练集比例
    print("\n【测试4】不同训练集比例的影响")
    print("-" * 60)
    for ratio in [0.1, 0.2, 0.3, 0.5]:
        results = attacker.run_complete_attack(train_ratio=ratio, model_type='rf')
        print(f"训练比例 {ratio:.1%}: 准确率 {results['metrics']['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    test_attribute_inference()

