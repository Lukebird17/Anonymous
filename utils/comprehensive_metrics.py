"""
完整的评估指标体系
包含去匿名化、属性推断、隐私保护等多维度指标
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, confusion_matrix
)
import json


class DeAnonymizationMetrics:
    """去匿名化攻击评估指标"""
    
    @staticmethod
    def accuracy(predictions: Dict[int, int], ground_truth: Dict[int, int]) -> float:
        """
        计算精确匹配准确率 (Matching Precision)
        
        Args:
            predictions: {orig_node: pred_anon_node}
            ground_truth: {orig_node: true_anon_node}
            
        Returns:
            准确率
        """
        correct = 0
        total = 0
        for orig_node, true_anon in ground_truth.items():
            if orig_node in predictions:
                if predictions[orig_node] == true_anon:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def precision_at_k(predictions: Dict[int, List[int]], 
                      ground_truth: Dict[int, int],
                      k: int = 5) -> float:
        """
        计算 Precision@K - 正确匹配在前K个候选中的比例
        
        Args:
            predictions: {orig_node: [top_k_candidates]}
            ground_truth: {orig_node: true_anon_node}
            k: 考虑前K个候选
            
        Returns:
            Precision@K
        """
        correct = 0
        total = 0
        
        for orig_node, true_anon in ground_truth.items():
            if orig_node in predictions:
                candidates = predictions[orig_node][:k]
                if true_anon in candidates:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(predictions: Dict[int, List[int]], 
                            ground_truth: Dict[int, int]) -> float:
        """
        计算 MRR (Mean Reciprocal Rank)
        
        MRR = (1/N) * Σ(1/rank_i), 其中 rank_i 是正确匹配的排名
        
        Args:
            predictions: {orig_node: [ranked_candidates]}
            ground_truth: {orig_node: true_anon_node}
            
        Returns:
            MRR值
        """
        reciprocal_ranks = []
        
        for orig_node, true_anon in ground_truth.items():
            if orig_node in predictions:
                candidates = predictions[orig_node]
                if true_anon in candidates:
                    rank = candidates.index(true_anon) + 1  # 排名从1开始
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def top_k_accuracy_curve(predictions: Dict[int, List[int]], 
                           ground_truth: Dict[int, int],
                           max_k: int = 20) -> Dict[int, float]:
        """
        计算 Top-K 准确率曲线
        
        Args:
            predictions: {orig_node: [ranked_candidates]}
            ground_truth: {orig_node: true_anon_node}
            max_k: 最大K值
            
        Returns:
            {k: accuracy@k} 字典
        """
        curve = {}
        for k in range(1, max_k + 1):
            curve[k] = DeAnonymizationMetrics.precision_at_k(predictions, ground_truth, k)
        return curve
    
    @staticmethod
    def calculate_all_metrics(predictions: Dict[int, List[int]], 
                            ground_truth: Dict[int, int]) -> Dict:
        """
        计算所有去匿名化指标
        
        Args:
            predictions: {orig_node: [ranked_candidates]}
            ground_truth: {orig_node: true_anon_node}
            
        Returns:
            所有指标的字典
        """
        # Top-1 准确率
        top1_preds = {node: candidates[0] for node, candidates in predictions.items() if candidates}
        accuracy = DeAnonymizationMetrics.accuracy(top1_preds, ground_truth)
        
        # Precision@K
        p_at_5 = DeAnonymizationMetrics.precision_at_k(predictions, ground_truth, k=5)
        p_at_10 = DeAnonymizationMetrics.precision_at_k(predictions, ground_truth, k=10)
        p_at_20 = DeAnonymizationMetrics.precision_at_k(predictions, ground_truth, k=20)
        
        # MRR
        mrr = DeAnonymizationMetrics.mean_reciprocal_rank(predictions, ground_truth)
        
        # Top-K曲线
        topk_curve = DeAnonymizationMetrics.top_k_accuracy_curve(predictions, ground_truth, max_k=20)
        
        # 随机基线
        n_nodes = len(ground_truth)
        random_baseline = 1.0 / n_nodes if n_nodes > 0 else 0.0
        
        # 相对改进倍数
        improvement_factor = accuracy / random_baseline if random_baseline > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision@5': p_at_5,
            'precision@10': p_at_10,
            'precision@20': p_at_20,
            'mrr': mrr,
            'topk_curve': topk_curve,
            'random_baseline': random_baseline,
            'improvement_factor': improvement_factor,
            'n_test_nodes': len(ground_truth)
        }


class AttributeInferenceMetrics:
    """属性推断攻击评估指标"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: List, y_pred: List) -> Dict:
        """
        计算分类指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            分类指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 每个类别的F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        return metrics


class RobustnessMetrics:
    """鲁棒性测试评估指标"""
    
    @staticmethod
    def calculate_robustness_curve(attack_results: Dict[float, Dict]) -> Dict:
        """
        计算鲁棒性曲线（攻击成功率 vs 图完整度）
        
        Args:
            attack_results: {completeness: attack_metrics} 字典
            
        Returns:
            鲁棒性曲线数据
        """
        curve = {
            'completeness': [],
            'accuracy': [],
            'precision@5': [],
            'precision@10': [],
            'mrr': []
        }
        
        for completeness in sorted(attack_results.keys(), reverse=True):
            metrics = attack_results[completeness]
            curve['completeness'].append(completeness)
            curve['accuracy'].append(metrics.get('accuracy', 0))
            curve['precision@5'].append(metrics.get('precision@5', 0))
            curve['precision@10'].append(metrics.get('precision@10', 0))
            curve['mrr'].append(metrics.get('mrr', 0))
        
        return curve
    
    @staticmethod
    def find_critical_point(robustness_curve: Dict, 
                          threshold: float = 0.5) -> Optional[float]:
        """
        找出攻击生效的"临界点" - 成功率下降到阈值时的完整度
        
        Args:
            robustness_curve: 鲁棒性曲线
            threshold: 成功率阈值（相对于完整图）
            
        Returns:
            临界完整度，如果找不到则返回None
        """
        if not robustness_curve['completeness']:
            return None
        
        # 完整图的准确率
        full_accuracy = robustness_curve['accuracy'][0]
        target_accuracy = full_accuracy * threshold
        
        # 查找临界点
        for comp, acc in zip(robustness_curve['completeness'], robustness_curve['accuracy']):
            if acc < target_accuracy:
                return comp
        
        return None


class PrivacyMetrics:
    """隐私保护评估指标"""
    
    @staticmethod
    def calculate_privacy_gain(attack_success_before: float,
                              attack_success_after: float) -> Dict:
        """
        计算隐私增益 - 防御后攻击成功率的下降
        
        Args:
            attack_success_before: 防御前的攻击成功率
            attack_success_after: 防御后的攻击成功率
            
        Returns:
            隐私增益指标
        """
        absolute_gain = attack_success_before - attack_success_after
        relative_gain = absolute_gain / attack_success_before if attack_success_before > 0 else 0
        
        return {
            'attack_success_before': attack_success_before,
            'attack_success_after': attack_success_after,
            'absolute_privacy_gain': absolute_gain,
            'relative_privacy_gain': relative_gain,
            'privacy_improvement_factor': (1 - relative_gain) if relative_gain < 1 else 0
        }
    
    @staticmethod
    def calculate_utility_loss(structural_loss: Dict, utility_metrics: Dict) -> Dict:
        """
        计算效用损失
        
        Args:
            structural_loss: 结构损失指标
            utility_metrics: 效用指标
            
        Returns:
            效用损失汇总
        """
        loss = {
            'edge_perturbation_ratio': structural_loss.get('l1_distance', 0),
            'degree_mae': structural_loss.get('degree_mae', 0),
            'clustering_diff': structural_loss.get('clustering_diff', 0),
            'avg_path_diff': structural_loss.get('avg_path_diff', 0)
        }
        
        # 任务效用保持率
        if 'modularity_preservation' in utility_metrics:
            loss['modularity_preservation'] = utility_metrics['modularity_preservation']
        if 'degree_centrality_correlation' in utility_metrics:
            loss['centrality_preservation'] = utility_metrics['degree_centrality_correlation']
        
        return loss
    
    @staticmethod
    def privacy_utility_tradeoff(privacy_gains: List[Dict],
                                utility_losses: List[Dict],
                                epsilon_values: List[float]) -> Dict:
        """
        分析隐私-效用权衡
        
        Args:
            privacy_gains: 不同ε下的隐私增益
            utility_losses: 不同ε下的效用损失
            epsilon_values: ε值列表
            
        Returns:
            权衡分析结果
        """
        tradeoff = {
            'epsilon': epsilon_values,
            'privacy_gain': [pg['relative_privacy_gain'] for pg in privacy_gains],
            'utility_loss': [ul.get('edge_perturbation_ratio', 0) for ul in utility_losses]
        }
        
        # 找出最佳平衡点（隐私增益 - 效用损失 最大）
        balance_scores = [pg - ul for pg, ul in zip(tradeoff['privacy_gain'], tradeoff['utility_loss'])]
        best_idx = np.argmax(balance_scores)
        tradeoff['best_epsilon'] = epsilon_values[best_idx]
        tradeoff['best_balance_score'] = balance_scores[best_idx]
        
        return tradeoff


class ComprehensiveEvaluator:
    """综合评估器 - 汇总所有实验结果"""
    
    def __init__(self):
        """初始化综合评估器"""
        self.results = {
            'stage1_identity': {},
            'stage1_attribute': {},
            'stage2_robustness': {},
            'stage3_defense': {}
        }
    
    def add_identity_deanonymization_results(self, method: str, metrics: Dict):
        """添加身份去匿名化结果"""
        self.results['stage1_identity'][method] = metrics
    
    def add_attribute_inference_results(self, method: str, metrics: Dict):
        """添加属性推断结果"""
        self.results['stage1_attribute'][method] = metrics
    
    def add_robustness_results(self, completeness: float, metrics: Dict):
        """添加鲁棒性测试结果"""
        self.results['stage2_robustness'][completeness] = metrics
    
    def add_defense_results(self, epsilon: float, privacy_metrics: Dict, utility_metrics: Dict):
        """添加防御结果"""
        self.results['stage3_defense'][epsilon] = {
            'privacy': privacy_metrics,
            'utility': utility_metrics
        }
    
    def generate_summary_report(self) -> str:
        """生成综合报告"""
        report = []
        report.append("="*80)
        report.append("社交网络'结构指纹'综合评估报告")
        report.append("="*80)
        
        # 阶段一：多维隐私攻击
        report.append("\n【阶段一】多维隐私攻击")
        report.append("-"*80)
        
        # 身份去匿名化
        report.append("\n1.1 身份去匿名化攻击")
        if self.results['stage1_identity']:
            for method, metrics in self.results['stage1_identity'].items():
                report.append(f"\n  方法: {method}")
                report.append(f"    - 准确率: {metrics.get('accuracy', 0):.2%}")
                report.append(f"    - Precision@5: {metrics.get('precision@5', 0):.2%}")
                report.append(f"    - Precision@10: {metrics.get('precision@10', 0):.2%}")
                report.append(f"    - MRR: {metrics.get('mrr', 0):.4f}")
                report.append(f"    - 改进倍数: {metrics.get('improvement_factor', 0):.2f}x")
        
        # 属性推断
        report.append("\n1.2 属性推断攻击")
        if self.results['stage1_attribute']:
            for method, metrics in self.results['stage1_attribute'].items():
                report.append(f"\n  方法: {method}")
                report.append(f"    - 准确率: {metrics.get('accuracy', 0):.2%}")
                report.append(f"    - F1 (macro): {metrics.get('f1_macro', 0):.4f}")
                report.append(f"    - F1 (weighted): {metrics.get('f1_weighted', 0):.4f}")
        
        # 阶段二：现实场景模拟
        report.append("\n\n【阶段二】现实场景模拟 - 鲁棒性测试")
        report.append("-"*80)
        if self.results['stage2_robustness']:
            robustness_curve = RobustnessMetrics.calculate_robustness_curve(
                self.results['stage2_robustness']
            )
            critical_point = RobustnessMetrics.find_critical_point(robustness_curve)
            
            report.append("\n不同图完整度下的攻击成功率:")
            for comp, acc in zip(robustness_curve['completeness'], robustness_curve['accuracy']):
                report.append(f"  - 完整度 {comp:.0%}: 准确率 {acc:.2%}")
            
            if critical_point:
                report.append(f"\n临界点: 图完整度 {critical_point:.0%} 时攻击成功率显著下降")
        
        # 阶段三：差分隐私防御
        report.append("\n\n【阶段三】差分隐私防御")
        report.append("-"*80)
        if self.results['stage3_defense']:
            report.append("\n不同隐私预算 (ε) 下的表现:")
            for epsilon in sorted(self.results['stage3_defense'].keys()):
                result = self.results['stage3_defense'][epsilon]
                privacy = result.get('privacy', {})
                utility = result.get('utility', {})
                
                report.append(f"\n  ε = {epsilon}:")
                if 'relative_privacy_gain' in privacy:
                    report.append(f"    - 隐私增益: {privacy['relative_privacy_gain']:.2%}")
                if 'edge_perturbation_ratio' in utility:
                    report.append(f"    - 边扰动比例: {utility['edge_perturbation_ratio']:.2%}")
                if 'modularity_preservation' in utility:
                    report.append(f"    - 模块度保持: {utility['modularity_preservation']:.2%}")
        
        report.append("\n" + "="*80)
        report.append("报告生成完毕")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, output_path: str):
        """保存结果到JSON文件"""
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_path}")


def test_metrics():
    """测试评估指标"""
    print("\n" + "="*60)
    print("测试评估指标模块")
    print("="*60)
    
    # 测试去匿名化指标
    print("\n【测试1】去匿名化指标")
    print("-" * 60)
    
    # 模拟预测结果
    predictions = {
        0: [10, 15, 20, 5, 8],
        1: [15, 10, 8, 20, 5],
        2: [20, 15, 10, 8, 5]
    }
    ground_truth = {0: 10, 1: 8, 2: 20}
    
    metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
    print(f"准确率: {metrics['accuracy']:.2%}")
    print(f"Precision@5: {metrics['precision@5']:.2%}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"改进倍数: {metrics['improvement_factor']:.2f}x")
    
    # 测试属性推断指标
    print("\n【测试2】属性推断指标")
    print("-" * 60)
    y_true = ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B']
    y_pred = ['A', 'B', 'B', 'C', 'B', 'A', 'A', 'B']
    
    attr_metrics = AttributeInferenceMetrics.calculate_classification_metrics(y_true, y_pred)
    print(f"准确率: {attr_metrics['accuracy']:.2%}")
    print(f"F1 (macro): {attr_metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {attr_metrics['f1_weighted']:.4f}")
    
    # 测试综合评估器
    print("\n【测试3】综合评估器")
    print("-" * 60)
    evaluator = ComprehensiveEvaluator()
    evaluator.add_identity_deanonymization_results("DeepWalk+Seed", metrics)
    evaluator.add_attribute_inference_results("RandomForest", attr_metrics)
    
    report = evaluator.generate_summary_report()
    print(report)
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    test_metrics()

