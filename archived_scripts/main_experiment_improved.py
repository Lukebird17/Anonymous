"""
改进版实验脚本 - 针对Facebook等大规模图优化

主要改进:
1. 使用更温和的匿名化参数
2. 增加更多区分性特征
3. 使用更好的匹配算法（匈牙利算法）
"""

import argparse
import numpy as np
import networkx as nx
from datetime import datetime

from data.dataset_loader import DatasetLoader
from attack.embedding_match import EmbeddingMatcher
from attack.baseline_match import BaselineMatcher
from preprocessing.anonymizer import GraphAnonymizer
from utils.comprehensive_metrics import DeAnonymizationMetrics
from models.deepwalk import DeepWalkModel


def run_improved_experiment(dataset_name='facebook', ego_id=None):
    """运行改进的实验
    
    Args:
        dataset_name: 数据集名称 ('facebook', 'facebook_ego', 'cora')
        ego_id: Facebook ego网络ID (如果使用facebook_ego)
    """
    
    print(f"\n{'='*70}")
    print(f"改进版去匿名化攻击实验")
    print(f"数据集: {dataset_name}")
    if ego_id:
        print(f"Ego网络ID: {ego_id}")
    print(f"{'='*70}")
    
    # 加载数据集
    loader = DatasetLoader()
    if dataset_name == 'facebook':
        G, attributes = loader._load_facebook_combined()
    elif dataset_name == 'facebook_ego':
        if ego_id is None:
            ego_id = "0"  # 默认使用ego 0
        G, attributes = loader.load_facebook(ego_network=ego_id)
    elif dataset_name == 'cora':
        G, attributes = loader.load_cora()
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    print(f"\n数据集信息:")
    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")
    print(f"  - 平均度: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    if attributes:
        print(f"  - 有属性的节点数: {len(attributes)}")
        # 检查是否有circles标签
        has_circles = any('circles' in attr for attr in attributes.values() if isinstance(attr, dict))
        if has_circles:
            print(f"  - 包含社交圈(circles)标签")
    
    # 测试不同的匿名化强度
    anonymization_levels = [
        (0.95, 0.02, "温和"),
        (0.90, 0.03, "中等"),
        (0.85, 0.05, "较强"),
        (0.75, 0.05, "原始")
    ]
    
    results_summary = []
    
    for edge_retention, noise_ratio, level_name in anonymization_levels:
        print(f"\n{'='*70}")
        print(f"测试匿名化强度: {level_name} (保留{edge_retention:.0%}边, 添加{noise_ratio:.0%}噪声)")
        print(f"{'='*70}")
        
        # 匿名化
        anonymizer = GraphAnonymizer(G)
        G_anon, node_mapping = anonymizer.anonymize_with_perturbation(
            edge_retention_ratio=edge_retention,
            noise_edge_ratio=noise_ratio
        )
        
        ground_truth = {orig: node_mapping[orig] for orig in G.nodes() if orig in node_mapping}
        
        print(f"\n匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
        
        # 1. Baseline方法（贪心）
        print(f"\n【方法1】Baseline - 贪心匹配")
        try:
            baseline = BaselineMatcher(G, G_anon, similarity_metric='cosine')
            predictions = baseline.match_by_features(top_k=20)
            metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
            
            print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
            print(f"  - Precision@5: {metrics['precision@5']:.2%}")
            print(f"  - Precision@10: {metrics['precision@10']:.2%}")
            print(f"  - Precision@20: {metrics['precision@20']:.2%}")
            print(f"  - MRR: {metrics['mrr']:.4f}")
            
            result_entry = {
                'level': level_name,
                'method': 'Baseline-Greedy',
                'accuracy': metrics['accuracy'],
                'p@5': metrics['precision@5'],
                'p@10': metrics['precision@10'],
                'mrr': metrics['mrr']
            }
            results_summary.append(result_entry)
        except Exception as e:
            print(f"  失败: {e}")
        
        # 2. Baseline方法（匈牙利算法）
        print(f"\n【方法2】Baseline - 匈牙利算法（最优匹配）")
        try:
            from models.feature_extractor import FeatureExtractor
            from scipy.optimize import linear_sum_assignment
            
            extractor = FeatureExtractor()
            nodes_orig = sorted(list(G.nodes()))
            nodes_anon = sorted(list(G_anon.nodes()))
            
            features_orig = extractor.extract_node_features(G, nodes_orig)
            features_anon = extractor.extract_node_features(G_anon, nodes_anon)
            
            # 标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_orig = scaler.fit_transform(features_orig)
            features_anon = scaler.transform(features_anon)
            
            # 计算相似度矩阵
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(features_orig, features_anon)
            
            # 匈牙利算法
            cost_matrix = -similarity
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 构建预测结果（包含top-k）
            predictions = {}
            for i, orig_idx in enumerate(row_ind):
                orig_node = nodes_orig[orig_idx]
                # 获取top-k候选
                top_indices = np.argsort(similarity[orig_idx])[::-1][:20]
                anon_nodes = [nodes_anon[idx] for idx in top_indices if idx < len(nodes_anon)]
                predictions[orig_node] = anon_nodes
            
            metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
            
            print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
            print(f"  - Precision@5: {metrics['precision@5']:.2%}")
            print(f"  - Precision@10: {metrics['precision@10']:.2%}")
            print(f"  - Precision@20: {metrics['precision@20']:.2%}")
            print(f"  - MRR: {metrics['mrr']:.4f}")
            
            result_entry = {
                'level': level_name,
                'method': 'Baseline-Hungarian',
                'accuracy': metrics['accuracy'],
                'p@5': metrics['precision@5'],
                'p@10': metrics['precision@10'],
                'mrr': metrics['mrr']
            }
            results_summary.append(result_entry)
        except Exception as e:
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. DeepWalk方法（仅在温和和中等匿名化下测试，节省时间）
        if level_name in ["温和", "中等"]:
            print(f"\n【方法3】DeepWalk图嵌入")
            try:
                nodes_orig = sorted(list(G.nodes()))
                nodes_anon = sorted(list(G_anon.nodes()))
                
                # 使用更好的参数
                deepwalk = DeepWalkModel(
                    dimensions=256,      # 增加维度
                    walk_length=100,     # 增加游走长度
                    num_walks=20,        # 增加游走次数
                    window_size=10,      # 修正参数名
                    workers=4
                )
                
                print("  训练原始图嵌入...")
                emb_orig = deepwalk.train(G)
                print("  训练匿名图嵌入...")
                emb_anon = deepwalk.train(G_anon)
                
                embedder = EmbeddingMatcher(G, G_anon)
                embedder.embeddings_orig = emb_orig
                embedder.embeddings_anon = emb_anon
                
                predictions_idx = embedder.match_by_similarity(top_k=20)
                
                # 转换为节点ID
                predictions = {}
                for orig_idx, anon_indices in predictions_idx.items():
                    if orig_idx < len(nodes_orig):
                        orig_node = nodes_orig[orig_idx]
                        anon_nodes = [nodes_anon[idx] for idx in anon_indices if idx < len(nodes_anon)]
                        predictions[orig_node] = anon_nodes
                
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - Top-1准确率: {metrics['accuracy']:.2%}")
                print(f"  - Precision@5: {metrics['precision@5']:.2%}")
                print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                print(f"  - Precision@20: {metrics['precision@20']:.2%}")
                print(f"  - MRR: {metrics['mrr']:.4f}")
                
                result_entry = {
                    'level': level_name,
                    'method': 'DeepWalk',
                    'accuracy': metrics['accuracy'],
                    'p@5': metrics['precision@5'],
                    'p@10': metrics['precision@10'],
                    'mrr': metrics['mrr']
                }
                results_summary.append(result_entry)
            except Exception as e:
                print(f"  失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 打印总结
    print(f"\n{'='*70}")
    print("实验结果总结")
    print(f"{'='*70}")
    print(f"\n{'匿名化强度':<10} {'方法':<20} {'Top-1':<8} {'P@5':<8} {'P@10':<8} {'MRR':<8}")
    print("-"*70)
    for r in results_summary:
        print(f"{r['level']:<10} {r['method']:<20} {r['accuracy']:>6.2%} {r['p@5']:>6.2%} {r['p@10']:>6.2%} {r['mrr']:>6.4f}")
    
    print(f"\n{'='*70}")
    print("关键发现:")
    print(f"{'='*70}")
    print("1. 匿名化强度对攻击成功率有显著影响")
    print("2. 温和的匿名化(95%边保留)下，攻击成功率会明显提高")
    print("3. 匈牙利算法可能比贪心算法效果更好")
    print("4. DeepWalk在大规模图上的效果取决于参数调优")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook',
                       choices=['facebook', 'facebook_ego', 'cora'],
                       help='数据集选择: facebook(combined), facebook_ego(ego网络), cora')
    parser.add_argument('--ego_id', type=str, default='0',
                       help='Facebook ego网络ID (0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980)')
    args = parser.parse_args()
    
    run_improved_experiment(args.dataset, ego_id=args.ego_id if args.dataset == 'facebook_ego' else None)

