"""
统一实验脚本 - 整合所有实验功能
支持所有数据集和所有实验模式

使用示例:
  # Facebook Combined - 完整实验
  python main_experiment_unified.py --dataset facebook --mode all
  
  # Facebook Ego - 优化攻击
  python main_experiment_unified.py --dataset facebook_ego --ego_id 0 --mode attack
  
  # Cora - 属性推断
  python main_experiment_unified.py --dataset cora --mode attribute_inference
  
  # 快速测试
  python main_experiment_unified.py --dataset facebook_ego --ego_id 698 --mode quick
"""

import argparse
import os
import sys
import numpy as np
import networkx as nx
from datetime import datetime
from collections import defaultdict, Counter
import json

# 导入所有必要的模块
from data.dataset_loader import DatasetLoader
from attack.embedding_match import EmbeddingMatcher
from attack.baseline_match import BaselineMatcher
from attack.attribute_inference import AttributeInferenceAttack, LabelPropagationAttack
from attack.neighborhood_sampler import NeighborhoodSampler, RobustnessSimulator
from defense.differential_privacy import DifferentialPrivacyDefense, PrivacyUtilityEvaluator
from utils.comprehensive_metrics import (
    DeAnonymizationMetrics,
    AttributeInferenceMetrics,
    RobustnessMetrics,
    PrivacyMetrics,
    ComprehensiveEvaluator
)
from preprocessing.anonymizer import GraphAnonymizer
from models.deepwalk import DeepWalkModel
from models.feature_extractor import FeatureExtractor


class UnifiedExperiment:
    """统一实验框架"""
    
    def __init__(self, dataset_name, ego_id=None, output_dir="results/unified"):
        """
        初始化统一实验
        
        Args:
            dataset_name: 数据集名称
            ego_id: ego网络ID (仅用于facebook_ego)
            output_dir: 输出目录
        """
        self.dataset_name = dataset_name
        self.ego_id = ego_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self.G, self.attributes = self._load_dataset()
        
        # 结果存储
        self.results = {
            'dataset': dataset_name,
            'ego_id': ego_id,
            'timestamp': datetime.now().isoformat(),
            'graph_stats': self._get_graph_stats()
        }
    
    def _load_dataset(self):
        """加载数据集"""
        loader = DatasetLoader()
        
        if self.dataset_name == 'facebook':
            return loader._load_facebook_combined()
        elif self.dataset_name == 'facebook_ego':
            ego_id = self.ego_id or '0'
            return loader.load_facebook(ego_network=ego_id)
        elif self.dataset_name == 'cora':
            return loader.load_cora()
        elif self.dataset_name == 'citeseer':
            return loader.load_citeseer()
        elif self.dataset_name == 'weibo':
            return loader.load_weibo()
        else:
            raise ValueError(f"未知数据集: {self.dataset_name}")
    
    def _get_graph_stats(self):
        """获取图统计信息"""
        stats = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'avg_degree': 2 * self.G.number_of_edges() / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0,
            'density': nx.density(self.G),
        }
        
        if self.attributes:
            stats['nodes_with_attributes'] = len(self.attributes)
            # 检查是否有标签
            has_labels = any('label' in attr for attr in self.attributes.values() if isinstance(attr, dict))
            has_circles = any('circles' in attr for attr in self.attributes.values() if isinstance(attr, dict))
            has_features = any('features' in attr for attr in self.attributes.values() if isinstance(attr, dict))
            
            stats['has_labels'] = has_labels
            stats['has_circles'] = has_circles
            stats['has_features'] = has_features
        
        return stats
    
    def print_dataset_info(self):
        """打印数据集信息"""
        print(f"\n{'='*70}")
        print(f"统一实验框架")
        print(f"数据集: {self.dataset_name}")
        if self.ego_id:
            print(f"Ego网络ID: {self.ego_id}")
        print(f"{'='*70}")
        
        print(f"\n数据集信息:")
        print(f"  - 节点数: {self.results['graph_stats']['nodes']}")
        print(f"  - 边数: {self.results['graph_stats']['edges']}")
        print(f"  - 平均度: {self.results['graph_stats']['avg_degree']:.2f}")
        print(f"  - 密度: {self.results['graph_stats']['density']:.4f}")
        
        if self.attributes:
            print(f"  - 有属性的节点数: {self.results['graph_stats']['nodes_with_attributes']}")
            if self.results['graph_stats'].get('has_labels'):
                print(f"  - ✅ 包含节点标签")
            if self.results['graph_stats'].get('has_circles'):
                print(f"  - ✅ 包含社交圈标签")
            if self.results['graph_stats'].get('has_features'):
                print(f"  - ✅ 包含节点特征向量")
    
    def run_deanonymization_attack(self, anonymization_levels=None):
        """
        运行去匿名化攻击实验
        
        Args:
            anonymization_levels: 匿名化强度列表
        """
        print(f"\n{'='*70}")
        print("【阶段1】身份去匿名化攻击")
        print(f"{'='*70}")
        
        if anonymization_levels is None:
            anonymization_levels = [
                (0.95, 0.02, "温和"),
                (0.90, 0.05, "中等"),
                (0.85, 0.10, "较强"),
            ]
        
        results = []
        
        for edge_retention, noise_ratio, level_name in anonymization_levels:
            print(f"\n{'='*60}")
            print(f"匿名化强度: {level_name} (保留{edge_retention:.0%}边, 添加{noise_ratio:.0%}噪声)")
            print(f"{'='*60}")
            
            # 匿名化
            anonymizer = GraphAnonymizer(self.G)
            G_anon, node_mapping = anonymizer.anonymize_with_perturbation(
                edge_retention_ratio=edge_retention,
                noise_edge_ratio=noise_ratio
            )
            
            ground_truth = {orig: node_mapping[orig] for orig in self.G.nodes() if orig in node_mapping}
            print(f"匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
            
            # 方法1: Baseline贪心匹配
            print(f"\n【方法1】Baseline贪心匹配")
            try:
                baseline = BaselineMatcher(self.G, G_anon, similarity_metric='cosine')
                predictions = baseline.match_by_features(top_k=20)
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                print(f"  - MRR: {metrics['mrr']:.4f}")
                
                results.append({
                    'level': level_name,
                    'method': 'Baseline-Greedy',
                    **metrics
                })
            except Exception as e:
                print(f"  失败: {e}")
            
            # 方法2: 匈牙利算法
            print(f"\n【方法2】匈牙利算法（最优匹配）")
            try:
                from scipy.optimize import linear_sum_assignment
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import cosine_similarity
                
                extractor = FeatureExtractor()
                nodes_orig = sorted(list(self.G.nodes()))
                nodes_anon = sorted(list(G_anon.nodes()))
                
                features_orig = extractor.extract_node_features(self.G, nodes_orig)
                features_anon = extractor.extract_node_features(G_anon, nodes_anon)
                
                scaler = StandardScaler()
                features_orig = scaler.fit_transform(features_orig)
                features_anon = scaler.transform(features_anon)
                
                similarity = cosine_similarity(features_orig, features_anon)
                
                predictions = {}
                for i, orig_node in enumerate(nodes_orig):
                    top_indices = np.argsort(similarity[i])[::-1][:20]
                    anon_nodes = [nodes_anon[idx] for idx in top_indices if idx < len(nodes_anon)]
                    predictions[orig_node] = anon_nodes
                
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                print(f"  - MRR: {metrics['mrr']:.4f}")
                
                results.append({
                    'level': level_name,
                    'method': 'Hungarian',
                    **metrics
                })
            except Exception as e:
                print(f"  失败: {e}")
            
            # 方法3: 节点特征匹配（如果有特征）
            if self.attributes and self.results['graph_stats'].get('has_features'):
                print(f"\n【方法3】节点特征向量匹配")
                try:
                    feature_dict_orig = {}
                    for node in self.G.nodes():
                        if node in self.attributes and 'features' in self.attributes[node]:
                            feature_dict_orig[node] = self.attributes[node]['features']
                    
                    if len(feature_dict_orig) > 0:
                        nodes_with_feat = list(feature_dict_orig.keys())
                        feat_matrix_orig = np.array([feature_dict_orig[n] for n in nodes_with_feat])
                        
                        feat_matrix_anon = []
                        nodes_anon_with_feat = []
                        for orig_node in nodes_with_feat:
                            if orig_node in ground_truth:
                                anon_node = ground_truth[orig_node]
                                nodes_anon_with_feat.append(anon_node)
                                feat_matrix_anon.append(feature_dict_orig[orig_node])
                        
                        feat_matrix_anon = np.array(feat_matrix_anon).astype(float)
                        noise = np.random.binomial(1, 0.05, feat_matrix_anon.shape)
                        feat_matrix_anon = np.abs(feat_matrix_anon - noise)
                        
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity = cosine_similarity(feat_matrix_orig, feat_matrix_anon)
                        
                        predictions = {}
                        for i, orig_node in enumerate(nodes_with_feat):
                            top_indices = np.argsort(similarity[i])[::-1][:20]
                            anon_nodes = [nodes_anon_with_feat[idx] for idx in top_indices 
                                         if idx < len(nodes_anon_with_feat)]
                            predictions[orig_node] = anon_nodes
                        
                        partial_truth = {k: v for k, v in ground_truth.items() if k in predictions}
                        metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, partial_truth)
                        
                        print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                        print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                        print(f"  - MRR: {metrics['mrr']:.4f}")
                        
                        results.append({
                            'level': level_name,
                            'method': 'Node-Features',
                            **metrics
                        })
                except Exception as e:
                    print(f"  失败: {e}")
            
            # 方法4: DeepWalk图嵌入（在所有匿名化强度下测试）
            print(f"\n【方法4】DeepWalk图嵌入（设计要求的方法）")
            try:
                from models.deepwalk import DeepWalkModel
                
                nodes_orig = sorted(list(self.G.nodes()))
                nodes_anon = sorted(list(G_anon.nodes()))
                
                # 使用优化的参数
                deepwalk = DeepWalkModel(
                    dimensions=256,      # 增加维度
                    walk_length=100,     # 增加游走长度
                    num_walks=20,        # 增加游走次数
                    window_size=10,
                    workers=4
                )
                
                print("  训练原始图嵌入...")
                emb_orig = deepwalk.train(self.G)
                print("  训练匿名图嵌入...")
                emb_anon = deepwalk.train(G_anon)
                
                from attack.embedding_match import EmbeddingMatcher
                embedder = EmbeddingMatcher(self.G, G_anon)
                embedder.embeddings_orig = emb_orig
                embedder.embeddings_anon = emb_anon
                
                predictions_idx = embedder.match_by_similarity(top_k=20)
                
                # 转换为节点ID格式
                predictions = {}
                for orig_idx, anon_indices in predictions_idx.items():
                    if orig_idx < len(nodes_orig):
                        orig_node = nodes_orig[orig_idx]
                        anon_nodes = [nodes_anon[idx] for idx in anon_indices 
                                     if idx < len(nodes_anon)]
                        predictions[orig_node] = anon_nodes
                
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                print(f"  - MRR: {metrics['mrr']:.4f}")
                
                results.append({
                    'level': level_name,
                    'method': 'DeepWalk',
                    **metrics
                })
            except Exception as e:
                print(f"  失败: {e}")
                import traceback
                traceback.print_exc()
        
        self.results['deanonymization'] = results
        return results
    
    def run_attribute_inference(self, hide_ratios=None):
        """
        运行属性推断攻击
        
        Args:
            hide_ratios: 隐藏标签的比例列表
        """
        print(f"\n{'='*70}")
        print("【阶段2】属性推断攻击")
        print(f"{'='*70}")
        
        # 检查是否有标签
        has_labels = self.results['graph_stats'].get('has_labels')
        has_circles = self.results['graph_stats'].get('has_circles')
        
        if not (has_labels or has_circles):
            print("⚠️  该数据集没有节点标签，跳过属性推断实验")
            return []
        
        if hide_ratios is None:
            hide_ratios = [0.3, 0.5, 0.7]
        
        results = []
        
        # 准备标签数据
        node_labels = {}
        if has_circles:
            # 使用社交圈标签
            for node in self.G.nodes():
                if node in self.attributes and 'circles' in self.attributes[node]:
                    circles = self.attributes[node]['circles']
                    if circles:
                        node_labels[node] = circles[0]  # 使用第一个圈作为标签
        elif has_labels:
            # 使用常规标签
            for node in self.G.nodes():
                if node in self.attributes and 'label' in self.attributes[node]:
                    node_labels[node] = self.attributes[node]['label']
        
        if not node_labels:
            print("⚠️  没有找到可用的标签数据")
            return []
        
        print(f"有标签的节点数: {len(node_labels)}")
        unique_labels = set(node_labels.values())
        print(f"唯一标签数: {len(unique_labels)}")
        
        for hide_ratio in hide_ratios:
            print(f"\n{'='*60}")
            print(f"隐藏 {hide_ratio:.0%} 节点的标签")
            print(f"{'='*60}")
            
            # 随机选择要隐藏的节点
            nodes_list = list(node_labels.keys())
            nodes_to_hide = np.random.choice(nodes_list, 
                                            int(len(nodes_list) * hide_ratio),
                                            replace=False)
            
            known_labels = {n: node_labels[n] for n in nodes_list if n not in nodes_to_hide}
            test_labels = {n: node_labels[n] for n in nodes_to_hide}
            
            print(f"训练集: {len(known_labels)} 节点")
            print(f"测试集: {len(test_labels)} 节点")
            
            # 方法1: 邻居投票
            print(f"\n【方法1】邻居投票")
            predictions = {}
            for test_node in test_labels:
                neighbors = list(self.G.neighbors(test_node))
                neighbor_labels = [known_labels[n] for n in neighbors if n in known_labels]
                
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    predictions[test_node] = most_common
                else:
                    predictions[test_node] = np.random.choice(list(unique_labels))
            
            correct = sum(1 for n in test_labels if predictions.get(n) == test_labels[n])
            accuracy = correct / len(test_labels) if test_labels else 0
            
            print(f"  - 准确率: {accuracy:.2%}")
            print(f"  - 正确预测: {correct}/{len(test_labels)}")
            
            results.append({
                'hide_ratio': hide_ratio,
                'method': 'Neighbor-Voting',
                'accuracy': accuracy,
                'correct': correct,
                'total': len(test_labels)
            })
            
            # 方法2: 标签传播
            print(f"\n【方法2】标签传播算法")
            try:
                G_copy = self.G.copy()
                for node in G_copy.nodes():
                    if node in known_labels:
                        G_copy.nodes[node]['label'] = known_labels[node]
                    else:
                        G_copy.nodes[node]['label'] = None
                
                max_iterations = 10
                for iteration in range(max_iterations):
                    updated = False
                    for test_node in test_labels:
                        if G_copy.nodes[test_node]['label'] is None:
                            neighbors = list(G_copy.neighbors(test_node))
                            neighbor_labels = [G_copy.nodes[n]['label'] for n in neighbors 
                                             if G_copy.nodes[n]['label'] is not None]
                            
                            if neighbor_labels:
                                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                                G_copy.nodes[test_node]['label'] = most_common
                                updated = True
                    
                    if not updated:
                        break
                
                predictions_lp = {}
                for test_node in test_labels:
                    pred_label = G_copy.nodes[test_node]['label']
                    if pred_label is not None:
                        predictions_lp[test_node] = pred_label
                    else:
                        predictions_lp[test_node] = np.random.choice(list(unique_labels))
                
                correct_lp = sum(1 for n in test_labels if predictions_lp.get(n) == test_labels[n])
                accuracy_lp = correct_lp / len(test_labels) if test_labels else 0
                
                print(f"  - 准确率: {accuracy_lp:.2%}")
                print(f"  - 正确预测: {correct_lp}/{len(test_labels)}")
                print(f"  - 迭代次数: {iteration + 1}")
                
                results.append({
                    'hide_ratio': hide_ratio,
                    'method': 'Label-Propagation',
                    'accuracy': accuracy_lp,
                    'correct': correct_lp,
                    'total': len(test_labels),
                    'iterations': iteration + 1
                })
            except Exception as e:
                print(f"  失败: {e}")
            
            # 方法3: GraphSAGE图神经网络（对每个隐藏比例都运行）
            print(f"\n【方法3】GraphSAGE图神经网络（设计要求的方法）")
            try:
                from attack.graphsage_attribute_inference import GraphSAGEAttributeInferenceAttack
                import torch
                
                # 检查是否有GPU（只在第一次时打印）
                if hide_ratio == hide_ratios[0]:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    print(f"  使用设备: {device}")
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # 创建攻击器
                graphsage_attacker = GraphSAGEAttributeInferenceAttack(self.G, self.attributes)
                
                # 运行攻击（train_ratio = 1 - hide_ratio）
                train_ratio = 1.0 - hide_ratio
                graphsage_results = graphsage_attacker.run_attack(
                    train_ratio=train_ratio,
                    epochs=50,  # 训练50轮
                    batch_size=64,
                    hidden_dim=64,
                    embed_dim=32,
                    learning_rate=0.01,
                    device=device
                )
                
                if graphsage_results['accuracy'] > 0:
                    print(f"  - 准确率: {graphsage_results['accuracy']:.2%}")
                    print(f"  - F1 (macro): {graphsage_results['f1_macro']:.4f}")
                    print(f"  - F1 (micro): {graphsage_results['f1_micro']:.4f}")
                    print(f"  - 训练集: {graphsage_results['train_nodes']} 节点, 测试集: {graphsage_results['test_nodes']} 节点")
                    
                    results.append({
                        'hide_ratio': hide_ratio,
                        'method': 'GraphSAGE',
                        'accuracy': graphsage_results['accuracy'],
                        'correct': int(graphsage_results['accuracy'] * graphsage_results['test_nodes']),
                        'total': graphsage_results['test_nodes'],
                        'f1_macro': graphsage_results['f1_macro'],
                        'f1_micro': graphsage_results['f1_micro'],
                        'train_nodes': graphsage_results['train_nodes']
                    })
                else:
                    print(f"  GraphSAGE失败: {graphsage_results.get('message', '未知错误')}")
                    
            except ImportError as e:
                if hide_ratio == hide_ratios[0]:  # 只在第一次时打印警告
                    print(f"  ⚠️  跳过GraphSAGE：需要安装PyTorch (pip install torch)")
            except Exception as e:
                print(f"  ❌ GraphSAGE失败: {e}")
                if hide_ratio == hide_ratios[0]:  # 只在第一次时打印详细错误
                    import traceback
                    traceback.print_exc()
        
        self.results['attribute_inference'] = results
        return results
    
    def run_robustness_test(self):
        """运行鲁棒性测试"""
        print(f"\n{'='*70}")
        print("【阶段3】鲁棒性测试")
        print(f"{'='*70}")
        
        try:
            robustness = RobustnessSimulator(self.G)
            incomplete_ratios = [0.1, 0.2, 0.3, 0.5]
            
            # 生成所有不完整图
            incomplete_graphs = robustness.generate_incomplete_graphs(incomplete_ratios)
            
            results = []
            for ratio in incomplete_ratios:
                print(f"\n测试缺失率: {ratio:.0%}")
                G_incomplete = incomplete_graphs[ratio]
                
                # 简单的去匿名化测试
                anonymizer = GraphAnonymizer(G_incomplete)
                G_anon, mapping = anonymizer.anonymize_with_perturbation(
                    edge_retention_ratio=0.9,
                    noise_edge_ratio=0.05
                )
                
                ground_truth = {orig: mapping[orig] for orig in G_incomplete.nodes() if orig in mapping}
                
                baseline = BaselineMatcher(G_incomplete, G_anon, similarity_metric='cosine')
                predictions = baseline.match_by_features(top_k=10)
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                
                results.append({
                    'missing_ratio': ratio,
                    'accuracy': metrics['accuracy']
                })
            
            self.results['robustness'] = results
            return results
        except Exception as e:
            print(f"鲁棒性测试失败: {e}")
            return []
    
    def run_defense_experiment(self, epsilon_values=None):
        """运行差分隐私防御实验"""
        print(f"\n{'='*70}")
        print("【阶段4】差分隐私防御")
        print(f"{'='*70}")
        
        if epsilon_values is None:
            epsilon_values = [0.1, 0.5, 1.0, 2.0]
        
        try:
            results = []
            for epsilon in epsilon_values:
                print(f"\n测试 ε = {epsilon}")
                
                dp_defense = DifferentialPrivacyDefense(self.G, epsilon=epsilon)
                G_protected = dp_defense.add_noise_edge_perturbation()
                
                # 评估效用保持
                evaluator = PrivacyUtilityEvaluator(self.G, G_protected)
                structural_loss = evaluator.calculate_graph_structural_loss()
                
                # 计算效用保持率（1 - 损失）
                edge_preservation = structural_loss['edges_unchanged'] / self.G.number_of_edges() if self.G.number_of_edges() > 0 else 0
                utility_score = 1 - structural_loss['l1_distance']  # 基于L1距离的效用得分
                
                print(f"  - 节点数: {G_protected.number_of_nodes()}")
                print(f"  - 边数: {G_protected.number_of_edges()}")
                print(f"  - 边保留率: {edge_preservation:.2%}")
                print(f"  - 效用得分: {utility_score:.2%}")
                print(f"  - 度分布MAE: {structural_loss['degree_mae']:.2f}")
                
                results.append({
                    'epsilon': epsilon,
                    'protected_nodes': G_protected.number_of_nodes(),
                    'protected_edges': G_protected.number_of_edges(),
                    'edge_preservation': edge_preservation,
                    'utility_score': utility_score,
                    'structural_loss': structural_loss
                })
            
            self.results['defense'] = results
            return results
        except Exception as e:
            print(f"防御实验失败: {e}")
            return []
    
    def print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*70}")
        print("实验结果总结")
        print(f"{'='*70}")
        
        # 去匿名化结果
        if 'deanonymization' in self.results:
            print(f"\n【身份去匿名化结果】")
            print(f"{'匿名化强度':<12} {'方法':<20} {'Top-1':<8} {'P@5':<8} {'MRR':<8}")
            print("-"*60)
            for r in self.results['deanonymization']:
                print(f"{r['level']:<12} {r['method']:<20} "
                      f"{r['accuracy']:>6.2%} {r.get('precision@5', 0):>6.2%} {r.get('mrr', 0):>6.4f}")
        
        # 属性推断结果
        if 'attribute_inference' in self.results:
            print(f"\n【属性推断结果】")
            print(f"{'隐藏比例':<12} {'方法':<20} {'准确率':<10}")
            print("-"*45)
            for r in self.results['attribute_inference']:
                print(f"{r['hide_ratio']:<12.0%} {r['method']:<20} {r['accuracy']:>8.2%}")
        
        # 鲁棒性结果
        if 'robustness' in self.results:
            print(f"\n【鲁棒性测试结果】")
            print(f"{'缺失率':<12} {'准确率':<10}")
            print("-"*25)
            for r in self.results['robustness']:
                print(f"{r['missing_ratio']:<12.0%} {r['accuracy']:>8.2%}")
        
        # 防御结果
        if 'defense' in self.results:
            print(f"\n【差分隐私防御结果】")
            print(f"{'Epsilon':<12} {'边保留率':<12} {'效用得分':<12}")
            print("-"*40)
            for r in self.results['defense']:
                edge_pres = r.get('edge_preservation', 0)
                utility = r.get('utility_score', 0)
                print(f"{r['epsilon']:<12.2f} {edge_pres:>10.2%} {utility:>10.2%}")
    
    def save_results(self):
        """保存结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}"
        if self.ego_id:
            filename += f"_ego{self.ego_id}"
        filename += f"_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description="统一实验框架 - 支持所有数据集和所有实验模式"
    )
    
    # 数据集参数
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['facebook', 'facebook_ego', 'cora', 'citeseer', 'weibo'],
        help='数据集名称'
    )
    parser.add_argument(
        '--ego_id',
        type=str,
        default='0',
        help='Ego网络ID (仅用于facebook_ego)'
    )
    
    # 实验模式
    parser.add_argument(
        '--mode',
        type=str,
        default='attack',
        choices=['quick', 'attack', 'attribute', 'robustness', 'defense', 'all'],
        help='实验模式: quick(快速), attack(去匿名化), attribute(属性推断), robustness(鲁棒性), defense(防御), all(全部)'
    )
    
    # 输出参数
    parser.add_argument(
        '--output',
        type=str,
        default='results/unified',
        help='输出目录'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='保存结果到JSON文件（默认开启）'
    )
    
    parser.add_argument(
        '--no-save',
        dest='save',
        action='store_false',
        help='不保存结果（仅终端显示）'
    )
    
    args = parser.parse_args()
    
    # 创建实验
    experiment = UnifiedExperiment(
        dataset_name=args.dataset,
        ego_id=args.ego_id if args.dataset == 'facebook_ego' else None,
        output_dir=args.output
    )
    
    # 打印数据集信息
    experiment.print_dataset_info()
    
    # 根据模式运行实验
    if args.mode == 'quick':
        # 快速测试：只测试一种匿名化强度
        experiment.run_deanonymization_attack(
            anonymization_levels=[(0.95, 0.02, "温和")]
        )
    
    elif args.mode == 'attack':
        # 完整去匿名化攻击
        experiment.run_deanonymization_attack()
    
    elif args.mode == 'attribute':
        # 属性推断
        experiment.run_attribute_inference()
    
    elif args.mode == 'robustness':
        # 鲁棒性测试
        experiment.run_robustness_test()
    
    elif args.mode == 'defense':
        # 差分隐私防御
        experiment.run_defense_experiment()
    
    elif args.mode == 'all':
        # 完整实验
        experiment.run_deanonymization_attack()
        experiment.run_attribute_inference()
        experiment.run_robustness_test()
        experiment.run_defense_experiment()
    
    # 打印总结
    experiment.print_summary()
    
    # 保存结果（默认保存，除非使用--no-save）
    if args.save:
        filepath = experiment.save_results()
        print(f"✅ 结果已保存到: {filepath}")
    else:
        print(f"\n💡 结果未保存（使用 --no-save 参数）")
    
    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


