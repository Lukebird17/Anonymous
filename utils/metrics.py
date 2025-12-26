"""
评估指标计算
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_accuracy(predictions: Dict[int, int], ground_truth: Dict[int, int]) -> float:
    """
    计算准确率
    
    Args:
        predictions: {匿名节点ID: 预测的原始节点ID}
        ground_truth: {匿名节点ID: 真实的原始节点ID}
    
    Returns:
        准确率
    """
    if not predictions:
        return 0.0
    
    correct = sum(1 for anon_id, pred_id in predictions.items() 
                  if ground_truth.get(anon_id) == pred_id)
    return correct / len(predictions)


def calculate_top_k_accuracy(similarity_matrix: np.ndarray, 
                             ground_truth_mapping: List[int],
                             k_values: List[int] = [1, 5, 10, 20]) -> Dict[int, float]:
    """
    计算Top-K准确率
    
    Args:
        similarity_matrix: 相似度矩阵 [n_anon_nodes, n_original_nodes]
        ground_truth_mapping: 真实映射，ground_truth_mapping[i]是第i个匿名节点对应的原始节点索引
        k_values: K值列表
    
    Returns:
        {k: top-k准确率}
    """
    n_nodes = similarity_matrix.shape[0]
    results = {}
    
    # 对每行（每个匿名节点）找到最相似的k个原始节点
    for k in k_values:
        top_k_predictions = np.argsort(-similarity_matrix, axis=1)[:, :k]
        
        correct = 0
        for i in range(n_nodes):
            if ground_truth_mapping[i] in top_k_predictions[i]:
                correct += 1
        
        results[k] = correct / n_nodes
    
    return results


def calculate_precision_recall_f1(predictions: Dict[int, int], 
                                   ground_truth: Dict[int, int]) -> Tuple[float, float, float]:
    """
    计算精确率、召回率和F1分数
    
    Args:
        predictions: {匿名节点ID: 预测的原始节点ID}
        ground_truth: {匿名节点ID: 真实的原始节点ID}
    
    Returns:
        (precision, recall, f1)
    """
    if not predictions:
        return 0.0, 0.0, 0.0
    
    # 真阳性：预测正确的数量
    true_positive = sum(1 for anon_id, pred_id in predictions.items() 
                       if ground_truth.get(anon_id) == pred_id)
    
    # 精确率 = TP / (TP + FP) = TP / 所有预测
    precision = true_positive / len(predictions) if predictions else 0.0
    
    # 召回率 = TP / (TP + FN) = TP / 所有真实样本
    recall = true_positive / len(ground_truth) if ground_truth else 0.0
    
    # F1分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def calculate_rank_metrics(similarity_matrix: np.ndarray,
                           ground_truth_mapping: List[int]) -> Dict[str, float]:
    """
    计算排名相关指标
    
    Args:
        similarity_matrix: 相似度矩阵
        ground_truth_mapping: 真实映射
    
    Returns:
        包含MRR和平均排名的字典
    """
    n_nodes = similarity_matrix.shape[0]
    ranks = []
    reciprocal_ranks = []
    
    for i in range(n_nodes):
        true_id = ground_truth_mapping[i]
        
        # 获取排序后的索引（降序）
        sorted_indices = np.argsort(-similarity_matrix[i])
        
        # 找到真实节点的排名
        rank = np.where(sorted_indices == true_id)[0][0] + 1  # 排名从1开始
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    
    return {
        "MRR": np.mean(reciprocal_ranks),  # Mean Reciprocal Rank
        "average_rank": np.mean(ranks),
        "median_rank": np.median(ranks)
    }


def print_evaluation_results(results: Dict):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
    """
    print("\n" + "="*50)
    print("去匿名化攻击评估结果")
    print("="*50)
    
    if "accuracy" in results:
        print(f"\n准确率: {results['accuracy']:.4f}")
    
    if "precision" in results:
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1']:.4f}")
    
    if "top_k" in results:
        print("\nTop-K准确率:")
        for k, acc in results["top_k"].items():
            print(f"  Top-{k}: {acc:.4f}")
    
    if "MRR" in results:
        print(f"\nMRR (Mean Reciprocal Rank): {results['MRR']:.4f}")
        print(f"平均排名: {results['average_rank']:.2f}")
        print(f"中位数排名: {results['median_rank']:.2f}")
    
    print("="*50 + "\n")


def compare_methods(results_dict: Dict[str, Dict]):
    """
    比较不同方法的结果
    
    Args:
        results_dict: {方法名: 评估结果}
    """
    print("\n" + "="*70)
    print("方法对比")
    print("="*70)
    
    # 表头
    print(f"{'方法':<15} {'准确率':<10} {'Top-5':<10} {'Top-10':<10} {'MRR':<10}")
    print("-"*70)
    
    # 每个方法的结果
    for method_name, results in results_dict.items():
        acc = results.get("accuracy", 0)
        top5 = results.get("top_k", {}).get(5, 0)
        top10 = results.get("top_k", {}).get(10, 0)
        mrr = results.get("MRR", 0)
        
        print(f"{method_name:<15} {acc:<10.4f} {top5:<10.4f} {top10:<10.4f} {mrr:<10.4f}")
    
    print("="*70 + "\n")


