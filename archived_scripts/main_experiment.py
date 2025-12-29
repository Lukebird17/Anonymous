"""
主实验脚本 - 完整的"结构指纹"三阶段实验
从攻击到防御的闭环研究

运行方式:
  python main_experiment.py --dataset facebook --mode all
  python main_experiment.py --dataset cora --mode attack
  python main_experiment.py --dataset weibo --mode defense
"""

import argparse
import os
import sys
import numpy as np
import networkx as nx
from datetime import datetime
import json

# 导入自定义模块
from data.dataset_loader import DatasetLoader
from attack.embedding_match import EmbeddingMatcher
from attack.baseline_match import BaselineMatcher
from attack.attribute_inference import AttributeInferenceAttack, LabelPropagationAttack
from attack.neighborhood_sampler import NeighborhoodSampler, RobustnessSimulator, LocalViewGenerator
from defense.differential_privacy import DifferentialPrivacyDefense, PrivacyUtilityEvaluator
from utils.comprehensive_metrics import (
    DeAnonymizationMetrics,
    AttributeInferenceMetrics,
    RobustnessMetrics,
    PrivacyMetrics,
    ComprehensiveEvaluator
)
from preprocessing.anonymizer import GraphAnonymizer


class StructuralFingerprintExperiment:
    """结构指纹完整实验"""
    
    def __init__(self, dataset_name: str, output_dir: str = "results/structural_fingerprint"):
        """
        初始化实验
        
        Args:
            dataset_name: 数据集名称
            output_dir: 输出目录
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        print(f"\n{'='*70}")
        print(f"加载数据集: {dataset_name}")
        print(f"{'='*70}")
        
        loader = DatasetLoader()
        if dataset_name == 'facebook':
            self.G, self.attributes = loader._load_facebook_combined()
        elif dataset_name == 'cora':
            self.G, self.attributes = loader.load_cora()
        elif dataset_name == 'citeseer':
            self.G, self.attributes = loader.load_citeseer()
        elif dataset_name == 'weibo':
            self.G, self.attributes = loader.load_weibo()
        else:
            raise ValueError(f"未知的数据集: {dataset_name}")
        
        # 检查数据集是否成功加载
        if self.G is None:
            raise RuntimeError(
                f"数据集 {dataset_name} 加载失败。\n"
                f"可能的原因：\n"
                f"  1. 网络连接问题（无法下载数据集）\n"
                f"  2. 数据集文件不存在\n"
                f"建议：\n"
                f"  - 检查网络连接\n"
                f"  - 手动下载数据集到 data/datasets/{dataset_name}/ 目录\n"
                f"  - 或使用其他数据集（如 cora, citeseer）"
            )
        
        # 综合评估器
        self.evaluator = ComprehensiveEvaluator()
        
        print(f"\n数据集加载完成:")
        print(f"  - 节点数: {self.G.number_of_nodes()}")
        print(f"  - 边数: {self.G.number_of_edges()}")
        print(f"  - 平均度: {2*self.G.number_of_edges()/self.G.number_of_nodes():.2f}")
    
    def stage1_identity_deanonymization(self):
        """
        阶段一：身份去匿名化攻击
        使用 DeepWalk + GraphSAGE 等方法
        """
        print(f"\n{'='*70}")
        print("【阶段一】身份去匿名化攻击")
        print(f"{'='*70}")
        
        # 1. 匿名化图
        print("\n步骤1: 对图进行匿名化处理...")
        anonymizer = GraphAnonymizer(self.G)
        G_anon, node_mapping = anonymizer.anonymize_with_perturbation(
            edge_retention_ratio=0.75,
            noise_edge_ratio=0.05
        )
        
        # 反向映射
        reverse_mapping = {v: k for k, v in node_mapping.items()}
        
        # 2. 基准方法 - 传统特征匹配
        print("\n步骤2: 运行基准方法（传统图特征匹配）...")
        try:
            baseline = BaselineMatcher(self.G, G_anon)
            predictions_baseline = baseline.match_by_features(top_k=10)
            
            # 构建ground truth
            ground_truth = {orig: node_mapping[orig] for orig in self.G.nodes() if orig in node_mapping}
            
            # 评估
            metrics_baseline = DeAnonymizationMetrics.calculate_all_metrics(predictions_baseline, ground_truth)
            print(f"\n基准方法结果:")
            print(f"  - 准确率: {metrics_baseline['accuracy']:.2%}")
            print(f"  - Precision@5: {metrics_baseline['precision@5']:.2%}")
            print(f"  - MRR: {metrics_baseline['mrr']:.4f}")
            
            self.evaluator.add_identity_deanonymization_results("Baseline", metrics_baseline)
        except Exception as e:
            import traceback
            print(f"基准方法失败: {e}")
            traceback.print_exc()
        
        # 3. DeepWalk 方法
        print("\n步骤3: 运行 DeepWalk 图嵌入方法...")
        embedder = None  # 初始化变量
        try:
            from models.deepwalk import DeepWalkModel
            
            # 获取节点列表（需要保持顺序一致）
            nodes_orig = sorted(list(self.G.nodes()))
            nodes_anon = sorted(list(G_anon.nodes()))
            
            # 训练嵌入
            deepwalk = DeepWalkModel(dimensions=128, walk_length=80, num_walks=10)
            emb_orig = deepwalk.train(self.G)
            emb_anon = deepwalk.train(G_anon)
            
            # 匹配
            embedder = EmbeddingMatcher(self.G, G_anon)
            embedder.embeddings_orig = emb_orig
            embedder.embeddings_anon = emb_anon
            
            # 获取基于索引的预测结果
            predictions_idx = embedder.match_by_similarity(top_k=10)
            
            # 转换为节点ID格式 {orig_node_id: [top-k anon_node_ids]}
            predictions_deepwalk = {}
            for orig_idx, anon_indices in predictions_idx.items():
                if orig_idx < len(nodes_orig):
                    orig_node = nodes_orig[orig_idx]
                    anon_nodes = [nodes_anon[idx] for idx in anon_indices if idx < len(nodes_anon)]
                    predictions_deepwalk[orig_node] = anon_nodes
            
            metrics_deepwalk = DeAnonymizationMetrics.calculate_all_metrics(predictions_deepwalk, ground_truth)
            
            print(f"\nDeepWalk 结果:")
            print(f"  - 准确率: {metrics_deepwalk['accuracy']:.2%}")
            print(f"  - Precision@10: {metrics_deepwalk['precision@10']:.2%}")
            
            self.evaluator.add_identity_deanonymization_results("DeepWalk", metrics_deepwalk)
        except Exception as e:
            import traceback
            print(f"DeepWalk 方法失败: {e}")
            traceback.print_exc()
        
        # 4. DeepWalk + 种子节点对齐
        print("\n步骤4: 运行 DeepWalk + 种子节点对齐...")
        try:
            if embedder is None:
                print("  跳过（DeepWalk未成功运行）")
                return G_anon, node_mapping
            
            # 获取节点列表
            nodes_orig = sorted(list(self.G.nodes()))
            nodes_anon = sorted(list(G_anon.nodes()))
            
            # 选择种子节点（5%）
            seed_ratio = 0.05
            all_nodes = list(ground_truth.keys())
            n_seeds = max(1, int(len(all_nodes) * seed_ratio))
            seed_nodes = np.random.choice(all_nodes, n_seeds, replace=False)
            
            # 转换为索引格式的种子映射
            seed_mapping_idx = {}
            for node in seed_nodes:
                if node in nodes_orig and ground_truth[node] in nodes_anon:
                    orig_idx = nodes_orig.index(node)
                    anon_idx = nodes_anon.index(ground_truth[node])
                    seed_mapping_idx[orig_idx] = anon_idx
            
            # 对齐
            predictions_idx = embedder.match_with_seeds(seed_mapping_idx, top_k=10)
            
            # 转换为节点ID格式
            predictions_aligned = {}
            for orig_idx, anon_indices in predictions_idx.items():
                if orig_idx < len(nodes_orig):
                    orig_node = nodes_orig[orig_idx]
                    anon_nodes = [nodes_anon[idx] for idx in anon_indices if idx < len(nodes_anon)]
                    predictions_aligned[orig_node] = anon_nodes
            
            metrics_aligned = DeAnonymizationMetrics.calculate_all_metrics(predictions_aligned, ground_truth)
            
            print(f"\nDeepWalk + 种子对齐结果 ({seed_ratio:.1%} 种子):")
            print(f"  - 准确率: {metrics_aligned['accuracy']:.2%}")
            print(f"  - Precision@10: {metrics_aligned['precision@10']:.2%}")
            
            self.evaluator.add_identity_deanonymization_results("DeepWalk+Seed", metrics_aligned)
        except Exception as e:
            import traceback
            print(f"种子对齐方法失败: {e}")
            traceback.print_exc()
        
        return G_anon, node_mapping
    
    def stage1_attribute_inference(self):
        """
        阶段一：属性推断攻击
        利用同质性原理推断节点隐藏属性
        """
        print(f"\n{'='*70}")
        print("【阶段一】属性推断攻击")
        print(f"{'='*70}")
        
        # 检查是否有标签属性
        has_labels = any('label' in attr for attr in self.attributes.values() if attr)
        
        if not has_labels:
            print("\n数据集没有标签属性，跳过属性推断攻击")
            return
        
        # 1. 基于结构特征的随机森林分类器
        print("\n方法1: 随机森林分类器 (结构特征 + 原始特征)")
        try:
            attacker = AttributeInferenceAttack(self.G, self.attributes)
            results_rf = attacker.run_complete_attack(train_ratio=0.3, model_type='rf')
            
            print(f"  - 准确率: {results_rf['metrics']['accuracy']:.2%}")
            print(f"  - F1 (macro): {results_rf['metrics']['f1_macro']:.4f}")
            
            self.evaluator.add_attribute_inference_results("RandomForest", results_rf['metrics'])
        except Exception as e:
            print(f"随机森林方法失败: {e}")
        
        # 2. 标签传播算法
        print("\n方法2: 标签传播算法")
        try:
            lp_attacker = LabelPropagationAttack(self.G, self.attributes)
            results_lp = lp_attacker.run_attack(train_ratio=0.3)
            
            print(f"  - 准确率: {results_lp['metrics']['accuracy']:.2%}")
            print(f"  - F1 (macro): {results_lp['metrics']['f1_macro']:.4f}")
            
            self.evaluator.add_attribute_inference_results("LabelPropagation", results_lp['metrics'])
        except Exception as e:
            print(f"标签传播方法失败: {e}")
    
    def stage2_robustness_test(self, G_anon: nx.Graph, node_mapping: dict):
        """
        阶段二：现实场景模拟 - 鲁棒性测试
        测试不同完整度下的攻击成功率
        """
        print(f"\n{'='*70}")
        print("【阶段二】现实场景模拟 - 鲁棒性测试")
        print(f"{'='*70}")
        
        # 测试不同的边缺失比例
        drop_ratios = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        robustness = RobustnessSimulator(G_anon)
        
        for drop_ratio in drop_ratios:
            completeness = 1.0 - drop_ratio
            print(f"\n测试完整度: {completeness:.0%} (缺失 {drop_ratio:.0%} 的边)")
            
            # 生成不完整图
            G_incomplete = robustness.drop_edges_random(drop_ratio)
            
            # 运行攻击
            try:
                baseline = BaselineMatcher(self.G, G_incomplete)
                predictions = baseline.match_by_features(top_k=10)
                
                # 构建ground truth
                ground_truth = {orig: node_mapping[orig] for orig in self.G.nodes() if orig in node_mapping}
                
                # 评估
                metrics = DeAnonymizationMetrics.calculate_all_metrics(predictions, ground_truth)
                
                print(f"  - 准确率: {metrics['accuracy']:.2%}")
                print(f"  - Precision@10: {metrics['precision@10']:.2%}")
                
                self.evaluator.add_robustness_results(completeness, metrics)
            except Exception as e:
                import traceback
                print(f"  攻击失败: {e}")
                traceback.print_exc()
        
        # 查找临界点
        robustness_curve = RobustnessMetrics.calculate_robustness_curve(
            self.evaluator.results['stage2_robustness']
        )
        critical_point = RobustnessMetrics.find_critical_point(robustness_curve, threshold=0.5)
        
        if critical_point:
            print(f"\n发现临界点: 当图完整度低于 {critical_point:.0%} 时，攻击成功率显著下降")
    
    def stage3_differential_privacy_defense(self):
        """
        阶段三：差分隐私防御
        测试不同隐私预算下的防御效果
        """
        print(f"\n{'='*70}")
        print("【阶段三】差分隐私防御")
        print(f"{'='*70}")
        
        # 测试不同的 epsilon 值
        epsilons = [0.5, 1.0, 2.0, 5.0]
        
        for epsilon in epsilons:
            print(f"\n测试隐私预算: ε = {epsilon}")
            print("-" * 70)
            
            # 应用差分隐私
            dp_defense = DifferentialPrivacyDefense(self.G, epsilon=epsilon)
            G_private = dp_defense.add_noise_edge_perturbation(seed=42)
            
            # 评估隐私保护效果
            try:
                # 在原图上运行攻击
                anonymizer = GraphAnonymizer(self.G)
                G_anon_orig, mapping_orig = anonymizer.anonymize_with_perturbation(
                    edge_retention_ratio=0.75, noise_edge_ratio=0.05
                )
                
                baseline_orig = BaselineMatcher(self.G, G_anon_orig)
                preds_orig = baseline_orig.match_by_features(top_k=10)
                gt_orig = {k: v for k, v in mapping_orig.items()}
                metrics_orig = DeAnonymizationMetrics.calculate_all_metrics(preds_orig, gt_orig)
                attack_success_before = metrics_orig['accuracy']
                
                # 在加噪后的图上运行攻击
                anonymizer_priv = GraphAnonymizer(G_private)
                G_anon_priv, mapping_priv = anonymizer_priv.anonymize_with_perturbation(
                    edge_retention_ratio=0.75, noise_edge_ratio=0.05
                )
                
                baseline_priv = BaselineMatcher(G_private, G_anon_priv)
                preds_priv = baseline_priv.match_by_features(top_k=10)
                gt_priv = {k: v for k, v in mapping_priv.items()}
                metrics_priv = DeAnonymizationMetrics.calculate_all_metrics(preds_priv, gt_priv)
                attack_success_after = metrics_priv['accuracy']
                
                # 计算隐私增益
                privacy_gain = PrivacyMetrics.calculate_privacy_gain(
                    attack_success_before, attack_success_after
                )
                
                print(f"\n隐私保护效果:")
                print(f"  - 攻击成功率 (防御前): {privacy_gain['attack_success_before']:.2%}")
                print(f"  - 攻击成功率 (防御后): {privacy_gain['attack_success_after']:.2%}")
                print(f"  - 隐私增益: {privacy_gain['relative_privacy_gain']:.2%}")
                
            except Exception as e:
                print(f"隐私评估失败: {e}")
                privacy_gain = {}
            
            # 评估效用损失
            evaluator = PrivacyUtilityEvaluator(self.G, G_private)
            structural_loss = evaluator.calculate_graph_structural_loss()
            utility_metrics = evaluator.evaluate_utility_for_tasks()
            
            utility_loss = PrivacyMetrics.calculate_utility_loss(structural_loss, utility_metrics)
            
            print(f"\n效用损失:")
            print(f"  - 边扰动比例: {utility_loss['edge_perturbation_ratio']:.2%}")
            print(f"  - 度数MAE: {utility_loss['degree_mae']:.2f}")
            print(f"  - 聚类系数差异: {utility_loss['clustering_diff']:.4f}")
            
            # 保存结果
            self.evaluator.add_defense_results(epsilon, privacy_gain, utility_loss)
    
    def run_complete_experiment(self, mode: str = 'all'):
        """
        运行完整实验
        
        Args:
            mode: 运行模式 ('all', 'attack', 'defense', 'robustness')
        """
        print(f"\n{'='*70}")
        print(f"开始运行完整实验 - 模式: {mode}")
        print(f"数据集: {self.dataset_name}")
        print(f"{'='*70}")
        
        start_time = datetime.now()
        
        # 阶段一：多维隐私攻击
        if mode in ['all', 'attack']:
            G_anon, node_mapping = self.stage1_identity_deanonymization()
            self.stage1_attribute_inference()
        else:
            # 创建简单的匿名化图用于后续测试
            anonymizer = GraphAnonymizer(self.G)
            G_anon, node_mapping = anonymizer.anonymize_with_perturbation(
                edge_retention_ratio=0.75, noise_edge_ratio=0.05
            )
        
        # 阶段二：现实场景模拟
        if mode in ['all', 'robustness']:
            self.stage2_robustness_test(G_anon, node_mapping)
        
        # 阶段三：差分隐私防御
        if mode in ['all', 'defense']:
            self.stage3_differential_privacy_defense()
        
        # 生成综合报告
        print(f"\n{'='*70}")
        print("生成综合评估报告")
        print(f"{'='*70}")
        
        report = self.evaluator.generate_summary_report()
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"{self.dataset_name}_{timestamp}_results.json")
        self.evaluator.save_results(results_file)
        
        report_file = os.path.join(self.output_dir, f"{self.dataset_name}_{timestamp}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存到: {report_file}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"\n实验完成！总耗时: {duration:.1f} 秒")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="社交网络结构指纹实验 - 从攻击到防御的闭环研究"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='facebook',
        choices=['facebook', 'cora', 'citeseer', 'weibo'],
        help='数据集名称'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'attack', 'defense', 'robustness'],
        help='运行模式'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/structural_fingerprint',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 运行实验
    experiment = StructuralFingerprintExperiment(
        dataset_name=args.dataset,
        output_dir=args.output
    )
    experiment.run_complete_experiment(mode=args.mode)


if __name__ == "__main__":
    main()

