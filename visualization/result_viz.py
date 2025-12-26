"""
结果可视化模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        self.style = 'seaborn-v0_8-darkgrid'
    
    def plot_confusion_matrix(self, similarity_matrix: np.ndarray,
                             ground_truth: np.ndarray,
                             output_path: Path = None):
        """绘制相似度混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix[:50, :50], cmap='YlOrRd', 
                   xticklabels=False, yticklabels=False)
        plt.xlabel('原始节点 (Original Nodes)', fontsize=12)
        plt.ylabel('匿名节点 (Anonymous Nodes)', fontsize=12)
        plt.title('节点相似度矩阵 (Node Similarity Matrix)', fontsize=14)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_seed_ratio_impact(self, results: Dict[float, Dict],
                              output_path: Path = None):
        """绘制种子节点比例的影响"""
        seed_ratios = sorted(results.keys())
        accuracies = [results[r]['accuracy'] for r in seed_ratios]
        
        plt.figure(figsize=(10, 6))
        plt.plot(seed_ratios, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('种子节点比例 (Seed Ratio)', fontsize=12)
        plt.ylabel('准确率 (Accuracy)', fontsize=12)
        plt.title('种子节点比例对攻击效果的影响', fontsize=14)
        plt.grid(alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"种子比例影响图已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_edge_retention_impact(self, results: Dict[float, Dict],
                                   output_path: Path = None):
        """绘制边保留率的影响"""
        retention_ratios = sorted(results.keys())
        accuracies = [results[r]['accuracy'] for r in retention_ratios]
        
        plt.figure(figsize=(10, 6))
        plt.plot(retention_ratios, accuracies, marker='s', linewidth=2, markersize=8)
        plt.xlabel('边保留率 (Edge Retention Ratio)', fontsize=12)
        plt.ylabel('准确率 (Accuracy)', fontsize=12)
        plt.title('边保留率对攻击效果的影响', fontsize=14)
        plt.grid(alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"边保留率影响图已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_summary_report(self, results: Dict, output_path: Path):
        """创建汇总报告"""
        report_lines = [
            "="*60,
            "去匿名化攻击实验报告",
            "="*60,
            "",
            "实验结果摘要:",
            ""
        ]
        
        for method, result in results.items():
            report_lines.append(f"\n方法: {method}")
            report_lines.append("-"*40)
            report_lines.append(f"  准确率: {result.get('accuracy', 0):.4f}")
            report_lines.append(f"  精确率: {result.get('precision', 0):.4f}")
            report_lines.append(f"  召回率: {result.get('recall', 0):.4f}")
            report_lines.append(f"  F1分数: {result.get('f1', 0):.4f}")
            
            if 'top_k' in result:
                report_lines.append("\n  Top-K准确率:")
                for k, acc in sorted(result['top_k'].items()):
                    report_lines.append(f"    Top-{k}: {acc:.4f}")
            
            if 'MRR' in result:
                report_lines.append(f"\n  MRR: {result['MRR']:.4f}")
                report_lines.append(f"  平均排名: {result['average_rank']:.2f}")
        
        report_lines.append("\n" + "="*60)
        
        # 保存报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"报告已保存: {output_path}")
        
        # 同时打印到控制台
        print('\n'.join(report_lines))


