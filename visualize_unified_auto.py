"""
自动化Unified实验结果可视化脚本
从 results/unified/ 目录读取JSON结果并生成图表
支持所有数据集（Cora, Facebook等）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
}


class UnifiedAutoVisualizer:
    """Unified实验结果自动可视化器"""
    
    def __init__(self, results_file=None):
        if results_file is None:
            results_file = self._find_latest_results()
        
        self.results_file = Path(results_file)
        self.output_dir = Path('results/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载结果
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # 提取数据集名称
        self.dataset_name = self.results.get('dataset', 'unknown')
        self.ego_id = self.results.get('ego_id', None)
        
        print(f"✓ 加载结果文件: {self.results_file}")
        print(f"✓ 数据集: {self.dataset_name}")
        if self.ego_id:
            print(f"✓ Ego ID: {self.ego_id}")
    
    def _find_latest_results(self):
        """查找最新的unified结果文件"""
        results_dir = Path('results/unified')
        json_files = list(results_dir.glob('*.json'))
        
        if not json_files:
            raise FileNotFoundError("未找到结果文件！请先运行 main_experiment_unified.py")
        
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("\n" + "="*70)
        print("生成可视化图表...")
        print("="*70)
        
        # 1. 去匿名化攻击
        if 'deanonymization' in self.results and self.results['deanonymization']:
            print("\n[图表 1] 去匿名化攻击性能")
            self.plot_deanonymization()
        
        # 2. 属性推断攻击
        if 'attribute_inference' in self.results and self.results['attribute_inference']:
            print("\n[图表 2] 属性推断攻击性能")
            self.plot_attribute_inference()
        
        # 3. 鲁棒性测试
        if 'robustness' in self.results and self.results['robustness']:
            print("\n[图表 3] 鲁棒性测试曲线")
            self.plot_robustness()
        
        # 4. 差分隐私防御
        if 'defense' in self.results and self.results['defense']:
            print("\n[图表 4] 差分隐私防御效果")
            self.plot_defense()
        
        # 5. 综合对比
        print("\n[图表 5] 综合实验分析")
        self.plot_comprehensive()
        
        print("\n" + "="*70)
        print(f"✅ 所有图表已生成！保存位置: {self.output_dir}")
        print("="*70)
        
        # 生成文本报告
        self.generate_text_report()
    
    def plot_deanonymization(self):
        """绘制去匿名化攻击结果"""
        data = self.results['deanonymization']
        
        # 按level和method组织数据
        levels = ['温和', '中等', '较强']
        methods = []
        for item in data:
            if item['method'] not in methods:
                methods.append(item['method'])
        
        # 显示所有方法（包括DeepWalk）
        main_methods = methods
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 准备数据
        data_dict = {}
        for item in data:
            key = f"{item['level']}-{item['method']}"
            data_dict[key] = item
        
        # 定义方法颜色
        method_colors = {
            'Baseline-Greedy': COLORS['primary'],
            'Hungarian': COLORS['secondary'],
            'Node-Features': COLORS['success'],
            'DeepWalk': COLORS['danger']
        }
        
        # 子图1: Top-1准确率对比
        ax1 = axes[0]
        x = np.arange(len(levels))
        width = 0.8 / len(main_methods)  # 动态调整宽度
        
        for i, method in enumerate(main_methods):
            accuracies = [data_dict[f'{level}-{method}']['accuracy'] * 100 
                         for level in levels if f'{level}-{method}' in data_dict]
            if accuracies:
                color = method_colors.get(method, list(COLORS.values())[i % len(COLORS)])
                ax1.bar(x + i*width - 0.4 + width/2, accuracies, width, label=method, color=color)
        
        ax1.set_xlabel('Anonymization Strength', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('De-anonymization Attack - Top-1 Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Mild', 'Medium', 'Strong'])
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: Precision@5对比
        ax2 = axes[1]
        for i, method in enumerate(main_methods):
            p5_scores = [data_dict[f'{level}-{method}']['precision@5'] * 100 
                        for level in levels if f'{level}-{method}' in data_dict]
            if p5_scores:
                color = method_colors.get(method, list(COLORS.values())[i % len(COLORS)])
                ax2.plot(['Mild', 'Medium', 'Strong'], p5_scores, 'o-', linewidth=2, markersize=8,
                        label=method, color=color)
        
        ax2.set_xlabel('Anonymization Strength', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision@5 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('De-anonymization Attack - Precision@5', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 子图3: MRR对比
        ax3 = axes[2]
        for i, method in enumerate(main_methods):
            mrr_scores = [data_dict[f'{level}-{method}']['mrr'] 
                         for level in levels if f'{level}-{method}' in data_dict]
            if mrr_scores:
                color = method_colors.get(method, list(COLORS.values())[i % len(COLORS)])
                ax3.plot(['Mild', 'Medium', 'Strong'], mrr_scores, 's-', linewidth=2, markersize=8,
                        label=method, color=color)
        
        ax3.set_xlabel('Anonymization Strength', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MRR', fontsize=12, fontweight='bold')
        ax3.set_title('De-anonymization Attack - MRR', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.dataset_name}_deanonymization.png'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_deanonymization.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path.name}")
    
    def plot_attribute_inference(self):
        """绘制属性推断攻击结果"""
        data = self.results['attribute_inference']
        
        # 按hide_ratio和method组织数据
        hide_ratios = sorted(list(set([item['hide_ratio'] for item in data])))
        methods = list(set([item['method'] for item in data]))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 准备数据
        data_dict = {}
        for item in data:
            key = f"{item['hide_ratio']}-{item['method']}"
            data_dict[key] = item
        
        # 定义方法颜色和显示名称
        method_config = {
            'Neighbor-Voting': {'color': COLORS['warning'], 'label': 'Neighbor Voting'},
            'Label-Propagation': {'color': COLORS['success'], 'label': 'Label Propagation'},
            'GraphSAGE': {'color': COLORS['danger'], 'label': 'GraphSAGE (GNN)'}
        }
        
        # 子图1: 准确率对比（柱状图）
        ax1 = axes[0]
        x = np.arange(len(hide_ratios))
        width = 0.8 / len(methods)  # 动态调整宽度
        
        hide_labels = [f'{int(r*100)}%' for r in hide_ratios]
        
        all_bars = []
        for i, method in enumerate(methods):
            if method in method_config:
                accuracies = [data_dict[f'{r}-{method}']['accuracy'] * 100 
                             for r in hide_ratios if f'{r}-{method}' in data_dict]
                
                if accuracies:
                    offset = (i - len(methods)/2 + 0.5) * width
                    bars = ax1.bar(x + offset, accuracies, width, 
                                  label=method_config[method]['label'],
                                  color=method_config[method]['color'], alpha=0.8)
                    all_bars.append(bars)
        
        # 添加数值标签
        for bars in all_bars:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Hidden Label Ratio', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Attribute Inference - Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(hide_labels)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 准确率趋势（折线图）
        ax2 = axes[1]
        x_numeric = [int(r*100) for r in hide_ratios]
        
        # 定义不同的标记样式
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, method in enumerate(methods):
            if method in method_config:
                accuracies = [data_dict[f'{r}-{method}']['accuracy'] * 100 
                             for r in hide_ratios if f'{r}-{method}' in data_dict]
                
                if accuracies:
                    marker = markers[i % len(markers)]
                    ax2.plot(x_numeric, accuracies, f'{marker}-', linewidth=3, markersize=10,
                            label=method_config[method]['label'], 
                            color=method_config[method]['color'])
        
        ax2.set_xlabel('Hidden Label Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Attribute Inference - Accuracy Trend', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'{self.dataset_name}_attribute_inference.png'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_attribute_inference.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path.name}")
    
    def plot_robustness(self):
        """绘制鲁棒性测试结果"""
        data = self.results['robustness']
        
        missing_ratios = sorted([item['missing_ratio'] for item in data])
        accuracies = [item['accuracy'] * 100 for item in sorted(data, key=lambda x: x['missing_ratio'])]
        
        completeness = [100 - r*100 for r in missing_ratios]
        
        # 计算相对下降
        baseline_acc = accuracies[0]
        relative_decline = [(baseline_acc - acc) / baseline_acc * 100 if baseline_acc > 0 else 0 
                           for acc in accuracies]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 攻击成功率曲线
        ax1 = axes[0]
        ax1.plot(completeness, accuracies, 'o-', linewidth=3, markersize=12,
                color=COLORS['primary'], label='Attack Accuracy')
        
        ax1.set_xlabel('Graph Completeness (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Attack Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Robustness Test - Attack Success Rate', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.invert_xaxis()  # 从高到低
        
        # 子图2: 相对下降率
        ax2 = axes[1]
        colors = [COLORS['success'] if d < 25 else COLORS['warning'] if d < 50 else COLORS['danger']
                 for d in relative_decline]
        
        bars = ax2.bar(completeness, relative_decline, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, relative_decline):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加50%下降线
        ax2.axhline(y=50, color='black', linestyle='--', linewidth=2,
                   label='50% Decline Threshold', alpha=0.5)
        
        ax2.set_xlabel('Graph Completeness (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Decline (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Robustness Test - Relative Decline', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.invert_xaxis()
        
        plt.tight_layout()
        filename = f'{self.dataset_name}_robustness.png'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_robustness.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path.name}")
    
    def plot_defense(self):
        """绘制差分隐私防御结果"""
        data = self.results['defense']
        
        epsilons = sorted([item['epsilon'] for item in data])
        edge_pres = [item['edge_preservation'] * 100 for item in sorted(data, key=lambda x: x['epsilon'])]
        utility_scores = [item['utility_score'] * 100 for item in sorted(data, key=lambda x: x['epsilon'])]
        degree_mae = [item['structural_loss']['degree_mae'] for item in sorted(data, key=lambda x: x['epsilon'])]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 边保留率
        ax1 = axes[0, 0]
        ax1.plot(epsilons, edge_pres, 'o-', linewidth=3, markersize=12,
                color=COLORS['success'])
        ax1.set_xlabel('Privacy Budget (epsilon)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Edge Preservation (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Differential Privacy - Edge Preservation', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(alpha=0.3)
        ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
        ax1.legend()
        
        # 子图2: 效用得分
        ax2 = axes[0, 1]
        ax2.plot(epsilons, utility_scores, 's-', linewidth=3, markersize=12,
                color=COLORS['primary'])
        ax2.set_xlabel('Privacy Budget (epsilon)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Utility Score (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Differential Privacy - Utility Score', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% Threshold')
        ax2.legend()
        
        # 子图3: 度分布MAE
        ax3 = axes[1, 0]
        ax3.plot(epsilons, degree_mae, '^-', linewidth=3, markersize=12,
                color=COLORS['danger'])
        ax3.set_xlabel('Privacy Budget (epsilon)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Degree Distribution MAE', fontsize=12, fontweight='bold')
        ax3.set_title('Differential Privacy - Degree Distribution Error', fontsize=14, fontweight='bold')
        ax3.set_xscale('log')
        ax3.grid(alpha=0.3)
        
        # 子图4: 隐私-效用权衡散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(utility_scores, edge_pres,
                            s=[1000/e for e in epsilons],
                            c=epsilons, cmap='viridis',
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        # 添加标签
        for i, eps in enumerate(epsilons):
            ax4.annotate(f'eps={eps}',
                       (utility_scores[i], edge_pres[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=11, fontweight='bold')
        
        # 添加参考线
        ax4.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=90, color='gray', linestyle='--', alpha=0.5)
        
        # 标注推荐值（ε=1.0）
        if 1.0 in epsilons:
            recommended_idx = epsilons.index(1.0)
            ax4.plot(utility_scores[recommended_idx], edge_pres[recommended_idx],
                    'r*', markersize=25, label='Recommended (eps=1.0)')
        
        ax4.set_xlabel('Utility Score (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Edge Preservation (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax4, label='ε')
        
        plt.tight_layout()
        filename = f'{self.dataset_name}_defense.png'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_defense.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path.name}")
    
    def plot_comprehensive(self):
        """绘制综合对比图"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 子图1: 所有攻击方法对比
        ax1 = fig.add_subplot(gs[0, :])
        
        methods_all = []
        accuracies_all = []
        colors_all = []
        
        # 去匿名化（温和匿名化下的最佳结果）
        if 'deanonymization' in self.results:
            mild_data = [item for item in self.results['deanonymization'] 
                        if item['level'] == '温和' and item['method'] != 'DeepWalk']
            for item in mild_data:
                methods_all.append(f"Identity-{item['method']}")
                accuracies_all.append(item['accuracy'] * 100)
                colors_all.append(COLORS['primary'])
        
        # 属性推断（30%隐藏）
        if 'attribute_inference' in self.results:
            attr_30 = [item for item in self.results['attribute_inference'] 
                      if item['hide_ratio'] == 0.3]
            for item in attr_30:
                methods_all.append(f"Attribute-{item['method']}")
                accuracies_all.append(item['accuracy'] * 100)
                colors_all.append(COLORS['secondary'])
        
        bars = ax1.barh(methods_all, accuracies_all, color=colors_all, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies_all):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {acc:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('[Comprehensive] All Attack Methods Performance (Best Case)',
                     fontsize=15, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3)
        
        # 子图2: 关键发现总结
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        # 计算关键统计数据
        summary_lines = ["[Key Findings Summary]\n"]
        
        if 'deanonymization' in self.results:
            best_deanon = max([item for item in self.results['deanonymization'] 
                              if item['level'] == '温和'],
                             key=lambda x: x['accuracy'])
            summary_lines.append(f"1. Identity De-anonymization:")
            summary_lines.append(f"   * Best Method: {best_deanon['method']} ({best_deanon['accuracy']*100:.2f}%)")
            summary_lines.append(f"   * Improvement: {best_deanon['improvement_factor']:.0f}x vs Random")
        
        if 'attribute_inference' in self.results:
            best_attr = max(self.results['attribute_inference'], 
                           key=lambda x: x['accuracy'])
            summary_lines.append(f"\n2. Attribute Inference:")
            summary_lines.append(f"   * Best Method: {best_attr['method']} ({best_attr['accuracy']*100:.2f}%)")
            summary_lines.append(f"   * Hidden Ratio: {best_attr['hide_ratio']*100:.0f}%")
        
        if 'robustness' in self.results:
            rob_data = sorted(self.results['robustness'], key=lambda x: x['missing_ratio'])
            baseline = rob_data[0]['accuracy'] * 100
            missing_ratios_str = ', '.join([f'{int(x["missing_ratio"]*100)}%' for x in rob_data])
            summary_lines.append(f"\n3. Robustness Test:")
            summary_lines.append(f"   * Baseline Accuracy: {baseline:.2f}%")
            summary_lines.append(f"   * Test Missing Ratios: {missing_ratios_str}")
        
        if 'defense' in self.results:
            summary_lines.append(f"\n4. Differential Privacy:")
            summary_lines.append(f"   * Recommended epsilon: 1.0")
            eps1_data = [item for item in self.results['defense'] if item['epsilon'] == 1.0]
            if eps1_data:
                summary_lines.append(f"   * Edge Preservation: {eps1_data[0]['edge_preservation']*100:.2f}%")
                summary_lines.append(f"   * Utility Score: {eps1_data[0]['utility_score']*100:.2f}%")
        
        summary_lines.append(f"\n[Summary]")
        summary_lines.append(f"Dataset {self.dataset_name} shows significant")
        summary_lines.append(f"privacy leakage risks under multi-dimensional")
        summary_lines.append(f"attacks. DP defense effectively mitigates risks!")
        
        summary_text = "\n".join(summary_lines)
        
        ax2.text(0.05, 0.95, summary_text,
                transform=ax2.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        filename = f'{self.dataset_name}_comprehensive.png'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_comprehensive.png'
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 已保存: {output_path.name}")
    
    def generate_text_report(self):
        """生成文本报告"""
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("实验结果总结报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据集: {self.dataset_name}")
        if self.ego_id:
            report_lines.append(f"Ego ID: {self.ego_id}")
        report_lines.append(f"结果文件: {self.results_file}")
        report_lines.append("="*70)
        
        # 图统计
        if 'graph_stats' in self.results:
            stats = self.results['graph_stats']
            report_lines.append("\n【图统计信息】")
            report_lines.append(f"  节点数: {stats['nodes']}")
            report_lines.append(f"  边数: {stats['edges']}")
            report_lines.append(f"  平均度: {stats['avg_degree']:.2f}")
            report_lines.append(f"  密度: {stats['density']:.6f}")
        
        # 去匿名化
        if 'deanonymization' in self.results:
            report_lines.append("\n【去匿名化攻击】")
            for item in self.results['deanonymization']:
                report_lines.append(f"\n{item['level']} - {item['method']}:")
                report_lines.append(f"  Top-1准确率: {item['accuracy']*100:.2f}%")
                report_lines.append(f"  Precision@5: {item['precision@5']*100:.2f}%")
                report_lines.append(f"  MRR: {item['mrr']:.4f}")
                report_lines.append(f"  提升倍数: {item['improvement_factor']:.0f}x")
        
        # 属性推断
        if 'attribute_inference' in self.results:
            report_lines.append("\n【属性推断攻击】")
            for item in self.results['attribute_inference']:
                report_lines.append(f"\n隐藏{item['hide_ratio']*100:.0f}% - {item['method']}:")
                report_lines.append(f"  准确率: {item['accuracy']*100:.2f}%")
                report_lines.append(f"  正确: {item['correct']}/{item['total']}")
        
        # 鲁棒性
        if 'robustness' in self.results:
            report_lines.append("\n【鲁棒性测试】")
            for item in self.results['robustness']:
                report_lines.append(f"缺失{item['missing_ratio']*100:.0f}%: 准确率 {item['accuracy']*100:.2f}%")
        
        # 防御
        if 'defense' in self.results:
            report_lines.append("\n【差分隐私防御】")
            for item in self.results['defense']:
                report_lines.append(f"\nε = {item['epsilon']}:")
                report_lines.append(f"  边保留率: {item['edge_preservation']*100:.2f}%")
                report_lines.append(f"  效用得分: {item['utility_score']*100:.2f}%")
                report_lines.append(f"  度分布MAE: {item['structural_loss']['degree_mae']:.2f}")
        
        report_lines.append("\n" + "="*70)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        filename = f'{self.dataset_name}_report.txt'
        if self.ego_id:
            filename = f'{self.dataset_name}_ego{self.ego_id}_report.txt'
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ 已生成文本报告: {output_path.name}")
        print("\n" + report_text)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Unified实验结果自动可视化")
    print("="*70)
    
    try:
        visualizer = UnifiedAutoVisualizer()
        visualizer.generate_all_figures()
        
        print("\n✅ 所有可视化任务完成！")
        print(f"\n📊 图表位置: {visualizer.output_dir}")
        print("\n生成的图表:")
        for fig_file in sorted(visualizer.output_dir.glob(f'{visualizer.dataset_name}*.png')):
            print(f"  - {fig_file.name}")
        
        return 0
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

