#!/usr/bin/env python3
"""
实验结果可视化脚本
生成论文级别的图表
"""

import matplotlib.pyplot as plt
import matplotlib
import json
import numpy as np
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置全局样式
plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

def load_results():
    """加载实验结果"""
    results_path = Path('results/attack_results.json')
    
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 转换为统一格式
            return {
                'baseline': {
                    'accuracy': data['基准方法(传统特征)']['accuracy'],
                    'top5_accuracy': data['基准方法(传统特征)']['top_k']['5'],
                    'top10_accuracy': data['基准方法(传统特征)']['top_k']['10']
                },
                'deepwalk': {
                    'accuracy': data['DeepWalk']['accuracy'],
                    'top5_accuracy': data['DeepWalk']['top_k']['5'],
                    'top10_accuracy': data['DeepWalk']['top_k']['10']
                },
                'deepwalk_seed': {
                    'accuracy': data['DeepWalk+种子(5%)']['accuracy'],
                    'top5_accuracy': data['DeepWalk+种子(5%)']['top_k']['5'],
                    'top10_accuracy': data['DeepWalk+种子(5%)']['top_k']['10']
                }
            }
    else:
        # 使用默认数据
        return {
            'baseline': {'accuracy': 0.0674, 'top5_accuracy': 0.2472, 'top10_accuracy': 0.3483},
            'deepwalk': {'accuracy': 0.0056, 'top5_accuracy': 0.0449, 'top10_accuracy': 0.0730},
            'deepwalk_seed': {'accuracy': 0.0730, 'top5_accuracy': 0.1461, 'top10_accuracy': 0.2247}
        }


def plot_accuracy_comparison(results, save_path):
    """图1: 准确率对比柱状图"""
    methods = ['基准方法\n(传统特征)', 'DeepWalk', 'DeepWalk+种子\n(5%)']
    accuracies = [
        results['baseline']['accuracy'] * 100,
        results['deepwalk']['accuracy'] * 100,
        results['deepwalk_seed']['accuracy'] * 100
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加随机基准线
    random_baseline = 100 / 178  # 1/178
    ax.axhline(y=random_baseline, color='red', linestyle='--', linewidth=2, 
               label=f'随机猜测 ({random_baseline:.2f}%)', alpha=0.7)
    
    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('去匿名化攻击准确率对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()


def plot_topk_comparison(results, save_path):
    """图2: Top-K准确率对比曲线"""
    k_values = [1, 5, 10]
    
    baseline_accs = [
        results['baseline']['accuracy'] * 100,
        results['baseline']['top5_accuracy'] * 100,
        results['baseline']['top10_accuracy'] * 100
    ]
    
    deepwalk_accs = [
        results['deepwalk']['accuracy'] * 100,
        results['deepwalk']['top5_accuracy'] * 100,
        results['deepwalk']['top10_accuracy'] * 100
    ]
    
    deepwalk_seed_accs = [
        results['deepwalk_seed']['accuracy'] * 100,
        results['deepwalk_seed']['top5_accuracy'] * 100,
        results['deepwalk_seed']['top10_accuracy'] * 100
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, baseline_accs, 'o-', linewidth=2.5, markersize=10,
            label='基准方法', color='#3498db')
    ax.plot(k_values, deepwalk_accs, 's-', linewidth=2.5, markersize=10,
            label='DeepWalk', color='#e74c3c')
    ax.plot(k_values, deepwalk_seed_accs, '^-', linewidth=2.5, markersize=10,
            label='DeepWalk+种子(5%)', color='#2ecc71')
    
    # 添加数值标签
    for k, b, d, ds in zip(k_values, baseline_accs, deepwalk_accs, deepwalk_seed_accs):
        ax.text(k, b + 1, f'{b:.1f}%', ha='center', fontsize=9)
        ax.text(k, d + 1, f'{d:.1f}%', ha='center', fontsize=9)
        ax.text(k, ds + 1, f'{ds:.1f}%', ha='center', fontsize=9)
    
    ax.set_xlabel('Top-K', fontsize=14, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Top-K准确率对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(k_values)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()


def plot_grouped_comparison(results, save_path):
    """图3: 分组柱状图 - 三种指标对比"""
    methods = ['基准方法', 'DeepWalk', 'DeepWalk+种子']
    
    top1 = [results['baseline']['accuracy'] * 100,
            results['deepwalk']['accuracy'] * 100,
            results['deepwalk_seed']['accuracy'] * 100]
    
    top5 = [results['baseline']['top5_accuracy'] * 100,
            results['deepwalk']['top5_accuracy'] * 100,
            results['deepwalk_seed']['top5_accuracy'] * 100]
    
    top10 = [results['baseline']['top10_accuracy'] * 100,
             results['deepwalk']['top10_accuracy'] * 100,
             results['deepwalk_seed']['top10_accuracy'] * 100]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, top1, width, label='Top-1', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, top5, width, label='Top-5', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, top10, width, label='Top-10', color='#2ecc71', alpha=0.8)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('不同方法在各Top-K指标下的表现', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()


def plot_improvement_analysis(save_path):
    """图4: 改进倍数分析"""
    random_baseline = 100 / 178  # 0.56%
    
    methods = ['随机猜测', '基准方法', 'DeepWalk', 'DeepWalk+种子']
    accuracies = [random_baseline, 6.74, 0.56, 7.30]
    improvements = [1, 6.74/random_baseline, 1, 7.30/random_baseline]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：准确率
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71']
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 绝对准确率', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    
    # 右图：改进倍数
    bars2 = ax2.bar(methods, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('相对随机猜测的倍数', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 相对改进倍数', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    
    plt.suptitle('攻击效果分析：绝对准确率 vs 相对改进', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()


def plot_data_statistics(save_path):
    """图5: 数据集统计"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图5.1: 节点和边数量
    categories = ['原始图', '匿名图']
    nodes = [178, 178]
    edges = [420, 315]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, nodes, width, label='节点数', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, edges, width, label='边数', color='#e74c3c', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('数量', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 图规模统计', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # 图5.2: 边保留率
    labels = ['保留的边', '删除的边']
    sizes = [315, 105]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('(b) 匿名化边保留情况', fontsize=13, fontweight='bold')
    
    # 图5.3: 平均度分布
    graphs = ['原始图', '匿名图']
    avg_degrees = [420*2/178, 315*2/178]
    
    bars = ax3.bar(graphs, avg_degrees, color=['#3498db', '#e74c3c'], alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('平均度', fontsize=12, fontweight='bold')
    ax3.set_title('(c) 平均度对比', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 图5.4: 实验配置
    configs = ['边保留率', '噪声边比例', '种子节点比例']
    values = [75, 5, 5]
    colors_cfg = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax4.barh(configs, values, color=colors_cfg, alpha=0.8)
    for bar in bars:
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width}%', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax4.set_xlabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) 实验参数配置', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('数据集与实验配置统计', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图5已保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("生成实验结果可视化图表")
    print("="*70)
    
    # 创建输出目录
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    print("\n📊 加载实验结果...")
    results = load_results()
    
    # 生成各种图表
    print("\n🎨 生成图表...")
    plot_accuracy_comparison(results, output_dir / 'fig1_accuracy_comparison.png')
    plot_topk_comparison(results, output_dir / 'fig2_topk_curves.png')
    plot_grouped_comparison(results, output_dir / 'fig3_grouped_comparison.png')
    plot_improvement_analysis(output_dir / 'fig4_improvement_analysis.png')
    plot_data_statistics(output_dir / 'fig5_data_statistics.png')
    
    print("\n" + "="*70)
    print("✅ 所有图表生成完成！")
    print("="*70)
    print(f"\n📁 图表保存位置: {output_dir}")
    print("\n生成的图表:")
    print("  1. fig1_accuracy_comparison.png  - 准确率对比柱状图")
    print("  2. fig2_topk_curves.png         - Top-K准确率曲线")
    print("  3. fig3_grouped_comparison.png  - 分组对比图")
    print("  4. fig4_improvement_analysis.png - 改进倍数分析")
    print("  5. fig5_data_statistics.png     - 数据集统计")
    
    print("\n💡 使用方法:")
    print("  - 在Finder中打开: open results/figures/")
    print("  - 或直接查看: open results/figures/fig1_accuracy_comparison.png")


if __name__ == "__main__":
    main()

