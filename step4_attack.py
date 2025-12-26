#!/usr/bin/env python3
"""
运行去匿名化攻击
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
from models.deepwalk import DeepWalk
from models.feature_extractor import FeatureExtractor
from attack.baseline_match import BaselineMatcher
from attack.embedding_match import EmbeddingMatcher
from attack.graph_alignment import GraphAligner
from utils.metrics import (calculate_accuracy, calculate_top_k_accuracy,
                           calculate_precision_recall_f1, print_evaluation_results,
                           compare_methods)

def load_data():
    """加载数据"""
    base_dir = Path(__file__).parent
    
    graph_path = base_dir / 'data' / 'processed' / 'graph.gpickle'
    anon_path = base_dir / 'data' / 'anonymized' / 'anonymous_graph.gpickle'
    gt_path = base_dir / 'data' / 'anonymized' / 'ground_truth.pkl'
    
    if not all([p.exists() for p in [graph_path, anon_path, gt_path]]):
        print("❌ 数据文件不完整，请先运行前面的步骤")
        return None, None, None
    
    with open(graph_path, 'rb') as f:
        G_orig = pickle.load(f)
    
    with open(anon_path, 'rb') as f:
        G_anon = pickle.load(f)
    
    with open(gt_path, 'rb') as f:
        ground_truth = pickle.load(f)
    
    return G_orig, G_anon, ground_truth


def attack_baseline(G_orig, G_anon, ground_truth):
    """基准攻击（传统特征）"""
    print("\n" + "="*60)
    print("方法1: 基准攻击（传统特征）")
    print("="*60)
    
    matcher = BaselineMatcher(similarity_metric='cosine')
    
    anon_nodes = sorted(G_anon.nodes())
    orig_nodes = sorted(G_orig.nodes())
    
    print("提取特征...")
    anon_features = matcher.extract_features(G_anon, anon_nodes)
    orig_features = matcher.extract_features(G_orig, orig_nodes)
    
    print("计算相似度...")
    similarity = matcher.compute_similarity_matrix(anon_features, orig_features)
    
    print("执行匹配...")
    predictions = matcher.match_greedy(similarity)
    
    # 构建ground truth映射
    gt_mapping = {}
    for i, anon_node in enumerate(anon_nodes):
        orig_node = ground_truth['reverse_mapping'][anon_node]
        orig_idx = orig_nodes.index(orig_node)
        gt_mapping[i] = orig_idx
    
    gt_list = [gt_mapping[i] for i in range(len(anon_nodes))]
    
    # 评估
    acc = calculate_accuracy(predictions, gt_mapping)
    p, r, f1 = calculate_precision_recall_f1(predictions, gt_mapping)
    top_k = calculate_top_k_accuracy(similarity, gt_list, [1, 5, 10, 20])
    
    results = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'top_k': top_k}
    print_evaluation_results(results)
    
    return results


def attack_deepwalk(G_orig, G_anon, ground_truth, use_seeds=False, seed_ratio=0.05):
    """DeepWalk攻击"""
    method_name = f"DeepWalk{'(+种子)' if use_seeds else ''}"
    print("\n" + "="*60)
    print(f"方法2: {method_name}")
    print("="*60)
    
    # 转换为无向图
    G_orig_u = G_orig.to_undirected() if G_orig.is_directed() else G_orig
    G_anon_u = G_anon.to_undirected() if G_anon.is_directed() else G_anon
    
    print("训练DeepWalk模型...")
    print("  原始图...")
    model_orig = DeepWalk(dimensions=128, walk_length=80, num_walks=10, epochs=5)
    model_orig.fit(G_orig_u)
    
    print("  匿名图...")
    model_anon = DeepWalk(dimensions=128, walk_length=80, num_walks=10, epochs=5)
    model_anon.fit(G_anon_u)
    
    # 获取嵌入
    anon_nodes = sorted(G_anon.nodes())
    orig_nodes = sorted(G_orig.nodes())
    
    anon_emb = model_anon.get_embeddings(anon_nodes)
    orig_emb = model_orig.get_embeddings(orig_nodes)
    
    # 构建ground truth
    gt_mapping = {}
    for i, anon_node in enumerate(anon_nodes):
        orig_node = ground_truth['reverse_mapping'][anon_node]
        orig_idx = orig_nodes.index(orig_node)
        gt_mapping[i] = orig_idx
    
    gt_list = [gt_mapping[i] for i in range(len(anon_nodes))]
    
    # 种子节点
    seed_pairs = []
    if use_seeds:
        n_seeds = int(len(anon_nodes) * seed_ratio)
        seed_indices = np.random.choice(len(anon_nodes), n_seeds, replace=False)
        seed_pairs = [(i, gt_mapping[i]) for i in seed_indices]
        print(f"\n使用 {len(seed_pairs)} 个种子节点 ({seed_ratio*100:.1f}%)")
        
        # 图对齐
        print("执行图对齐...")
        aligner = GraphAligner()
        anon_emb = aligner.align_procrustes(anon_emb, orig_emb, seed_pairs)
    
    # 匹配
    print("执行匹配...")
    matcher = EmbeddingMatcher()
    similarity = matcher.compute_similarity_matrix(anon_emb, orig_emb)
    
    if seed_pairs:
        predictions = matcher.match_with_seeds(similarity, seed_pairs)
    else:
        predictions = matcher.match_greedy(similarity)
    
    # 评估
    acc = calculate_accuracy(predictions, gt_mapping)
    p, r, f1 = calculate_precision_recall_f1(predictions, gt_mapping)
    top_k = calculate_top_k_accuracy(similarity, gt_list, [1, 5, 10, 20])
    
    results = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'top_k': top_k}
    print_evaluation_results(results)
    
    return results


def main():
    print("="*60)
    print("步骤4: 去匿名化攻击实验")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    G_orig, G_anon, ground_truth = load_data()
    
    if G_orig is None:
        return
    
    print(f"✅ 数据加载完成")
    print(f"   原始图: {G_orig.number_of_nodes()} 节点, {G_orig.number_of_edges()} 边")
    print(f"   匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
    
    # 运行攻击
    all_results = {}
    
    # 1. 基准方法
    results_baseline = attack_baseline(G_orig, G_anon, ground_truth)
    all_results['基准方法(传统特征)'] = results_baseline
    
    # 2. DeepWalk（无种子）
    results_dw = attack_deepwalk(G_orig, G_anon, ground_truth, use_seeds=False)
    all_results['DeepWalk'] = results_dw
    
    # 3. DeepWalk + 种子
    results_dw_seed = attack_deepwalk(G_orig, G_anon, ground_truth, 
                                      use_seeds=True, seed_ratio=0.05)
    all_results['DeepWalk+种子(5%)'] = results_dw_seed
    
    # 比较结果
    print("\n" + "="*60)
    compare_methods(all_results)
    
    # 保存结果
    import json
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'attack_results.json', 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型
        serializable = {}
        for method, result in all_results.items():
            serializable[method] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items()
            }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {results_dir}/attack_results.json")
    print(f"\n🎉 实验完成!")
    print(f"\n💡 关键发现:")
    print(f"   1. 即使删除{(1-ground_truth['edge_retention_ratio'])*100:.0f}%的边，")
    print(f"      攻击准确率仍可达 {results_dw_seed['accuracy']*100:.1f}%")
    print(f"   2. 使用5%种子节点后，准确率从 {results_dw['accuracy']*100:.1f}% 提升到 {results_dw_seed['accuracy']*100:.1f}%")
    print(f"   3. 这证明了'即便我不说话，我的朋友也会暴露我'！")


if __name__ == "__main__":
    main()


