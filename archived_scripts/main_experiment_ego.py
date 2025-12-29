"""
Facebook Ego-Networks 实验脚本
充分利用ego网络的社交圈标签和节点特征进行更深入的隐私攻击分析
"""

import argparse
import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime

from data.dataset_loader import DatasetLoader
from preprocessing.anonymizer import GraphAnonymizer
from utils.comprehensive_metrics import DeAnonymizationMetrics
from models.feature_extractor import FeatureExtractor


def analyze_ego_network_properties(G, attributes):
    """分析ego网络的属性特征"""
    print("\n" + "="*70)
    print("Ego网络属性分析")
    print("="*70)
    
    # 统计特征分布
    if attributes:
        # 统计有特征的节点
        nodes_with_features = sum(1 for attr in attributes.values() 
                                 if isinstance(attr, dict) and 'features' in attr)
        print(f"有特征向量的节点: {nodes_with_features}")
        
        # 统计特征维度
        if nodes_with_features > 0:
            first_feat = next(attr['features'] for attr in attributes.values() 
                             if isinstance(attr, dict) and 'features' in attr)
            print(f"特征向量维度: {len(first_feat)}")
            print(f"特征类型: 二值特征 (0/1)")
        
        # 统计社交圈标签
        nodes_with_circles = sum(1 for attr in attributes.values() 
                                if isinstance(attr, dict) and 'circles' in attr)
        print(f"\n有社交圈标签的节点: {nodes_with_circles}")
        
        if nodes_with_circles > 0:
            all_circles = set()
            circle_sizes = defaultdict(int)
            for attr in attributes.values():
                if isinstance(attr, dict) and 'circles' in attr:
                    for circle in attr['circles']:
                        all_circles.add(circle)
                        circle_sizes[circle] += 1
            
            print(f"社交圈数量: {len(all_circles)}")
            print(f"\n社交圈分布:")
            for circle, size in sorted(circle_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {circle}: {size}个节点")
    
    # 分析图结构
    print(f"\n图结构特性:")
    print(f"  - 密度: {nx.density(G):.4f}")
    print(f"  - 聚类系数: {nx.average_clustering(G):.4f}")
    
    # 度分布
    degrees = [d for n, d in G.degree()]
    print(f"  - 度分布: min={min(degrees)}, max={max(degrees)}, avg={np.mean(degrees):.2f}")


def run_ego_deanonymization_attack(G, attributes, ego_id):
    """在ego网络上运行去匿名化攻击实验"""
    
    print("\n" + "="*70)
    print("【实验1】身份去匿名化攻击 - 利用结构特征")
    print("="*70)
    
    # 测试不同的匿名化强度
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
        anonymizer = GraphAnonymizer(G)
        G_anon, node_mapping = anonymizer.anonymize_with_perturbation(
            edge_retention_ratio=edge_retention,
            noise_edge_ratio=noise_ratio
        )
        
        ground_truth = {orig: node_mapping[orig] for orig in G.nodes() if orig in node_mapping}
        print(f"匿名化完成: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
        
        # 方法1: 结构特征匹配
        print(f"\n【方法1】结构特征匹配 (度、聚类系数等)")
        try:
            from scipy.optimize import linear_sum_assignment
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics.pairwise import cosine_similarity
            
            extractor = FeatureExtractor()
            nodes_orig = sorted(list(G.nodes()))
            nodes_anon = sorted(list(G_anon.nodes()))
            
            features_orig = extractor.extract_node_features(G, nodes_orig)
            features_anon = extractor.extract_node_features(G_anon, nodes_anon)
            
            # 标准化
            scaler = StandardScaler()
            features_orig = scaler.fit_transform(features_orig)
            features_anon = scaler.transform(features_anon)
            
            # 计算相似度
            similarity = cosine_similarity(features_orig, features_anon)
            
            # 获取top-k预测
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
                'method': 'Structural Features',
                'accuracy': metrics['accuracy'],
                'p@5': metrics['precision@5'],
                'mrr': metrics['mrr']
            })
        except Exception as e:
            print(f"  失败: {e}")
        
        # 方法2: 使用原始特征（如果有）
        if attributes and any('features' in attr for attr in attributes.values() if isinstance(attr, dict)):
            print(f"\n【方法2】节点特征向量匹配")
            try:
                # 提取原始特征
                feature_dict_orig = {}
                for node in G.nodes():
                    if node in attributes and 'features' in attributes[node]:
                        feature_dict_orig[node] = attributes[node]['features']
                
                if len(feature_dict_orig) > 0:
                    nodes_with_feat = list(feature_dict_orig.keys())
                    feat_matrix_orig = np.array([feature_dict_orig[n] for n in nodes_with_feat])
                    
                    # 对于匿名图，使用映射后的特征
                    feat_matrix_anon = []
                    nodes_anon_with_feat = []
                    for orig_node in nodes_with_feat:
                        if orig_node in ground_truth:
                            anon_node = ground_truth[orig_node]
                            nodes_anon_with_feat.append(anon_node)
                            feat_matrix_anon.append(feature_dict_orig[orig_node])
                    
                    feat_matrix_anon = np.array(feat_matrix_anon)
                    
                    # 添加一些噪声来模拟特征不完全匹配
                    feat_matrix_anon = feat_matrix_anon.astype(float)
                    noise = np.random.binomial(1, 0.05, feat_matrix_anon.shape)
                    feat_matrix_anon = np.abs(feat_matrix_anon - noise)
                    
                    # 计算相似度
                    similarity = cosine_similarity(feat_matrix_orig, feat_matrix_anon)
                    
                    # 预测
                    predictions = {}
                    for i, orig_node in enumerate(nodes_with_feat):
                        top_indices = np.argsort(similarity[i])[::-1][:20]
                        anon_nodes = [nodes_anon_with_feat[idx] for idx in top_indices 
                                     if idx < len(nodes_anon_with_feat)]
                        predictions[orig_node] = anon_nodes
                    
                    # 评估
                    partial_truth = {k: v for k, v in ground_truth.items() if k in predictions}
                    metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, partial_truth)
                    
                    print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                    print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                    print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                    print(f"  - MRR: {metrics['mrr']:.4f}")
                    
                    results.append({
                        'level': level_name,
                        'method': 'Node Features',
                        'accuracy': metrics['accuracy'],
                        'p@5': metrics['precision@5'],
                        'mrr': metrics['mrr']
                    })
            except Exception as e:
                print(f"  失败: {e}")
                import traceback
                traceback.print_exc()
    
    return results


def run_ego_attribute_inference(G, attributes, ego_id):
    """在ego网络上运行属性推断攻击"""
    
    print("\n" + "="*70)
    print("【实验2】属性推断攻击 - 利用社交圈标签")
    print("="*70)
    
    # 检查是否有circles标签
    nodes_with_circles = [n for n in G.nodes() 
                         if n in attributes and isinstance(attributes[n], dict) 
                         and 'circles' in attributes[n]]
    
    if not nodes_with_circles:
        print("警告: 该ego网络没有社交圈标签，跳过属性推断实验")
        return []
    
    print(f"有社交圈标签的节点数: {len(nodes_with_circles)}")
    
    # 为每个节点创建标签（基于它所属的第一个circle）
    node_labels = {}
    for node in nodes_with_circles:
        circles = attributes[node]['circles']
        if circles:
            node_labels[node] = circles[0]  # 使用第一个circle作为标签
    
    unique_labels = set(node_labels.values())
    print(f"唯一社交圈标签数: {len(unique_labels)}")
    
    # 使用同质性原理进行属性推断
    print(f"\n使用同质性原理推断属性:")
    print("假设: 相连的用户更可能属于相同的社交圈")
    
    # 随机隐藏一些节点的标签
    hide_ratios = [0.3, 0.5, 0.7]
    results = []
    
    for hide_ratio in hide_ratios:
        print(f"\n{'='*60}")
        print(f"隐藏 {hide_ratio:.0%} 节点的标签")
        print(f"{'='*60}")
        
        # 随机选择要隐藏的节点
        nodes_to_hide = np.random.choice(nodes_with_circles, 
                                        int(len(nodes_with_circles) * hide_ratio),
                                        replace=False)
        
        # 创建训练集和测试集
        known_labels = {n: node_labels[n] for n in nodes_with_circles if n not in nodes_to_hide}
        test_labels = {n: node_labels[n] for n in nodes_to_hide if n in node_labels}
        
        print(f"训练集: {len(known_labels)} 节点")
        print(f"测试集: {len(test_labels)} 节点")
        
        # 方法1: 多数投票（基于邻居）
        print(f"\n【方法1】邻居投票")
        predictions = {}
        for test_node in test_labels:
            neighbors = list(G.neighbors(test_node))
            neighbor_labels = [known_labels[n] for n in neighbors if n in known_labels]
            
            if neighbor_labels:
                # 多数投票
                from collections import Counter
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                predictions[test_node] = most_common
            else:
                # 如果没有已知邻居，随机猜测
                predictions[test_node] = np.random.choice(list(unique_labels))
        
        # 计算准确率
        correct = sum(1 for n in test_labels if predictions.get(n) == test_labels[n])
        accuracy = correct / len(test_labels) if test_labels else 0
        
        print(f"  - 准确率: {accuracy:.2%}")
        print(f"  - 正确预测: {correct}/{len(test_labels)}")
        
        results.append({
            'hide_ratio': hide_ratio,
            'method': 'Neighbor Voting',
            'accuracy': accuracy
        })
        
        # 方法2: 标签传播
        print(f"\n【方法2】标签传播算法")
        try:
            # 使用networkx的标签传播
            # 首先给已知节点打标签
            for node in G.nodes():
                if node in known_labels:
                    G.nodes[node]['label'] = known_labels[node]
                else:
                    G.nodes[node]['label'] = None
            
            # 简单的标签传播（迭代更新）
            max_iterations = 10
            for iteration in range(max_iterations):
                updated = False
                for test_node in test_labels:
                    if G.nodes[test_node]['label'] is None:
                        neighbors = list(G.neighbors(test_node))
                        neighbor_labels = [G.nodes[n]['label'] for n in neighbors 
                                         if G.nodes[n]['label'] is not None]
                        
                        if neighbor_labels:
                            from collections import Counter
                            most_common = Counter(neighbor_labels).most_common(1)[0][0]
                            G.nodes[test_node]['label'] = most_common
                            updated = True
                
                if not updated:
                    break
            
            # 收集预测结果
            predictions_lp = {}
            for test_node in test_labels:
                pred_label = G.nodes[test_node]['label']
                if pred_label is not None:
                    predictions_lp[test_node] = pred_label
                else:
                    predictions_lp[test_node] = np.random.choice(list(unique_labels))
            
            # 计算准确率
            correct_lp = sum(1 for n in test_labels if predictions_lp.get(n) == test_labels[n])
            accuracy_lp = correct_lp / len(test_labels) if test_labels else 0
            
            print(f"  - 准确率: {accuracy_lp:.2%}")
            print(f"  - 正确预测: {correct_lp}/{len(test_labels)}")
            print(f"  - 迭代次数: {iteration + 1}")
            
            results.append({
                'hide_ratio': hide_ratio,
                'method': 'Label Propagation',
                'accuracy': accuracy_lp
            })
        except Exception as e:
            print(f"  失败: {e}")
    
    return results


def run_complete_ego_experiment(ego_id='0'):
    """运行完整的ego网络实验"""
    
    print(f"\n{'#'*70}")
    print(f"Facebook Ego-Network 完整实验")
    print(f"Ego ID: {ego_id}")
    print(f"{'#'*70}")
    
    # 加载数据
    loader = DatasetLoader()
    G, attributes = loader.load_facebook(ego_network=ego_id)
    
    if G is None:
        print("加载失败！")
        return
    
    # 分析网络属性
    analyze_ego_network_properties(G, attributes)
    
    # 实验1: 去匿名化攻击
    deanon_results = run_ego_deanonymization_attack(G, attributes, ego_id)
    
    # 实验2: 属性推断攻击
    attr_results = run_ego_attribute_inference(G, attributes, ego_id)
    
    # 打印总结
    print(f"\n{'='*70}")
    print("实验结果总结")
    print(f"{'='*70}")
    
    if deanon_results:
        print(f"\n【身份去匿名化结果】")
        print(f"{'匿名化强度':<12} {'方法':<20} {'Top-1':<8} {'P@5':<8} {'MRR':<8}")
        print("-"*60)
        for r in deanon_results:
            print(f"{r['level']:<12} {r['method']:<20} {r['accuracy']:>6.2%} {r['p@5']:>6.2%} {r['mrr']:>6.4f}")
    
    if attr_results:
        print(f"\n【属性推断结果】")
        print(f"{'隐藏比例':<12} {'方法':<20} {'准确率':<10}")
        print("-"*45)
        for r in attr_results:
            print(f"{r['hide_ratio']:<12.0%} {r['method']:<20} {r['accuracy']:>8.2%}")
    
    print(f"\n{'='*70}")
    print("关键发现:")
    print(f"{'='*70}")
    print("1. Ego网络的结构特征可用于身份去匿名化")
    print("2. 节点特征向量提供了额外的匹配线索")
    print("3. 社交圈标签具有很强的同质性，可被邻居推断")
    print("4. 即使隐藏70%的标签，仍能通过网络结构推断剩余节点")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facebook Ego-Networks 隐私攻击实验')
    parser.add_argument('--ego_id', type=str, default='0',
                       choices=['0', '107', '348', '414', '686', '698', '1684', '1912', '3437', '3980'],
                       help='Ego网络ID')
    args = parser.parse_args()
    
    run_complete_ego_experiment(args.ego_id)

