"""
完整的去匿名化攻击实验脚本
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import logging
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepwalk import DeepWalk
from models.feature_extractor import FeatureExtractor
from attack.baseline_match import BaselineMatcher
from attack.embedding_match import EmbeddingMatcher
from attack.graph_alignment import GraphAligner
from utils.metrics import (calculate_accuracy, calculate_top_k_accuracy,
                           calculate_precision_recall_f1, calculate_rank_metrics,
                           print_evaluation_results, compare_methods)
from visualization.result_viz import ResultVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline_attack(G_orig, G_anon, ground_truth):
    """运行基准攻击（传统特征）"""
    logger.info("\n" + "="*60)
    logger.info("运行基准攻击（传统特征）")
    logger.info("="*60)
    
    matcher = BaselineMatcher(similarity_metric='cosine')
    
    # 提取特征
    anon_nodes = sorted(G_anon.nodes())
    orig_nodes = sorted(G_orig.nodes())
    
    anon_features = matcher.extract_features(G_anon, anon_nodes)
    orig_features = matcher.extract_features(G_orig, orig_nodes)
    
    # 计算相似度
    similarity_matrix = matcher.compute_similarity_matrix(anon_features, orig_features)
    
    # 贪心匹配
    predictions = matcher.match_greedy(similarity_matrix)
    
    # 构建ground truth映射
    node_mapping = ground_truth['node_mapping']
    gt_mapping = {}
    for i, anon_node in enumerate(anon_nodes):
        orig_node = ground_truth['reverse_mapping'][anon_node]
        orig_idx = orig_nodes.index(orig_node)
        gt_mapping[i] = orig_idx
    
    # 计算ground truth列表（用于Top-K）
    gt_list = [gt_mapping[i] for i in range(len(anon_nodes))]
    
    # 评估
    accuracy = calculate_accuracy(predictions, gt_mapping)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, gt_mapping)
    top_k = calculate_top_k_accuracy(similarity_matrix, gt_list, k_values=[1, 5, 10, 20])
    rank_metrics = calculate_rank_metrics(similarity_matrix, gt_list)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top_k': top_k,
        **rank_metrics
    }
    
    print_evaluation_results(results)
    
    return results, similarity_matrix


def run_deepwalk_attack(G_orig, G_anon, ground_truth, use_alignment=False, seed_ratio=0.0):
    """运行DeepWalk攻击"""
    logger.info("\n" + "="*60)
    logger.info(f"运行DeepWalk攻击 (对齐={use_alignment}, 种子比例={seed_ratio})")
    logger.info("="*60)
    
    # 转换为无向图
    G_orig_u = G_orig.to_undirected() if G_orig.is_directed() else G_orig
    G_anon_u = G_anon.to_undirected() if G_anon.is_directed() else G_anon
    
    # 训练DeepWalk
    logger.info("训练原始图的DeepWalk...")
    model_orig = DeepWalk(dimensions=128, walk_length=80, num_walks=10)
    model_orig.fit(G_orig_u)
    
    logger.info("训练匿名图的DeepWalk...")
    model_anon = DeepWalk(dimensions=128, walk_length=80, num_walks=10)
    model_anon.fit(G_anon_u)
    
    # 获取嵌入
    anon_nodes = sorted(G_anon.nodes())
    orig_nodes = sorted(G_orig.nodes())
    
    anon_embeddings = model_anon.get_embeddings(anon_nodes)
    orig_embeddings = model_orig.get_embeddings(orig_nodes)
    
    # 构建ground truth映射
    node_mapping = ground_truth['node_mapping']
    gt_mapping = {}
    for i, anon_node in enumerate(anon_nodes):
        orig_node = ground_truth['reverse_mapping'][anon_node]
        orig_idx = orig_nodes.index(orig_node)
        gt_mapping[i] = orig_idx
    
    gt_list = [gt_mapping[i] for i in range(len(anon_nodes))]
    
    # 生成种子节点
    seed_pairs = []
    if seed_ratio > 0:
        n_seeds = int(len(anon_nodes) * seed_ratio)
        seed_indices = np.random.choice(len(anon_nodes), n_seeds, replace=False)
        seed_pairs = [(i, gt_mapping[i]) for i in seed_indices]
        logger.info(f"使用 {len(seed_pairs)} 个种子节点")
    
    # 图对齐
    if use_alignment and len(seed_pairs) > 0:
        aligner = GraphAligner()
        anon_embeddings = aligner.align_procrustes(anon_embeddings, orig_embeddings, seed_pairs)
    
    # 匹配
    matcher = EmbeddingMatcher()
    similarity_matrix = matcher.compute_similarity_matrix(anon_embeddings, orig_embeddings)
    
    if len(seed_pairs) > 0:
        predictions = matcher.match_with_seeds(similarity_matrix, seed_pairs)
    else:
        predictions = matcher.match_greedy(similarity_matrix)
    
    # 评估
    accuracy = calculate_accuracy(predictions, gt_mapping)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, gt_mapping)
    top_k = calculate_top_k_accuracy(similarity_matrix, gt_list, k_values=[1, 5, 10, 20])
    rank_metrics = calculate_rank_metrics(similarity_matrix, gt_list)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top_k': top_k,
        **rank_metrics
    }
    
    print_evaluation_results(results)
    
    return results, similarity_matrix


def main():
    parser = argparse.ArgumentParser(description='去匿名化攻击实验')
    parser.add_argument('--data', type=str, default='github',
                       help='数据集名称 (github/weibo)')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'baseline', 'deepwalk', 'deepwalk_seed'],
                       help='攻击方法')
    parser.add_argument('--seed_ratio', type=float, default=0.05,
                       help='种子节点比例')
    args = parser.parse_args()
    
    # 加载数据
    data_dir = Path(__file__).parent.parent
    original_path = data_dir / f"data/processed/{args.data}_graph.gpickle"
    anon_path = data_dir / "data/anonymized/anonymous_graph.gpickle"
    ground_truth_path = data_dir / "data/anonymized/ground_truth.pkl"
    
    if not all([p.exists() for p in [original_path, anon_path, ground_truth_path]]):
        logger.error("数据文件不存在，请先运行数据预处理和匿名化")
        return
    
    logger.info("加载数据...")
    with open(original_path, 'rb') as f:
        G_orig = pickle.load(f)
    
    with open(anon_path, 'rb') as f:
        G_anon = pickle.load(f)
    
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    
    logger.info(f"原始图: {G_orig.number_of_nodes()} 节点, {G_orig.number_of_edges()} 边")
    logger.info(f"匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
    
    # 运行攻击
    all_results = {}
    
    if args.method in ['all', 'baseline']:
        results, sim_matrix = run_baseline_attack(G_orig, G_anon, ground_truth)
        all_results['Baseline (传统特征)'] = results
    
    if args.method in ['all', 'deepwalk']:
        results, sim_matrix = run_deepwalk_attack(G_orig, G_anon, ground_truth,
                                                  use_alignment=False, seed_ratio=0.0)
        all_results['DeepWalk (无对齐)'] = results
    
    if args.method in ['all', 'deepwalk_seed']:
        results, sim_matrix = run_deepwalk_attack(G_orig, G_anon, ground_truth,
                                                  use_alignment=True, seed_ratio=args.seed_ratio)
        all_results[f'DeepWalk (种子={args.seed_ratio})'] = results
    
    # 比较结果
    if len(all_results) > 1:
        compare_methods(all_results)
    
    # 保存结果
    results_dir = data_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    import json
    with open(results_dir / "attack_results.json", 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python类型
        serializable_results = {}
        for method, result in all_results.items():
            serializable_results[method] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items()
            }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {results_dir / 'attack_results.json'}")
    
    # 生成可视化
    visualizer = ResultVisualizer()
    visualizer.create_summary_report(all_results, results_dir / "attack_report.txt")


if __name__ == "__main__":
    main()


