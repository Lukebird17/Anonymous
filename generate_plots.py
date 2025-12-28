#!/usr/bin/env python3
"""
实验结果可视化脚本
自动识别 JSON 结果并生成对比图表
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
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                print("⚠️ JSON 文件格式错误")
                return {}
    else:
        print("⚠️ 未找到结果文件，使用模拟数据")
        return {
            '基准方法': {'accuracy': 0.0674, 'top_k': {'1': 0.0674, '5': 0.2472, '10': 0.3483}},
            'DeepWalk': {'accuracy': 0.0056, 'top_k': {'1': 0.0056, '5': 0.0449, '10': 0.0730}},
            'DeepWalk+种子': {'accuracy': 0.0730, 'top_k': {'1': 0.0730, '5': 0.1461, '10': 0.2247}}
        }


def get_colors(n):
    """根据条目数量生成颜色列表"""
    cmap = plt.get_cmap('tab10')  # 使用 tab10 色板
    return [cmap(i) for i in range(n)]


def plot_accuracy_comparison(results, save_path):
    """图1: 准确率对比柱状图 (Top-1)"""
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in methods]

    colors = get_colors(len(methods))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 随机基准线 (假设178个用户)
    random_baseline = 100 / 178
    ax.axhline(y=random_baseline, color='red', linestyle='--', linewidth=2,
               label=f'随机猜测 ({random_baseline:.2f}%)', alpha=0.7)

    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('去匿名化攻击准确率对比 (Top-1)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {save_path}")
    plt.close()


def plot_topk_comparison(results, save_path):
    """图2: Top-K准确率对比曲线"""
    methods = list(results.keys())
    colors = get_colors(len(methods))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    fig, ax = plt.subplots(figsize=(12, 7))

    # 动态获取 K 值列表
    first_method = methods[0]
    k_keys = sorted(results[first_method]['top_k'].keys(), key=lambda x: int(x))
    k_values = [int(k) for k in k_keys]

    for idx, method in enumerate(methods):
        # 提取该方法对应的 Top-K 数据
        accs = []
        for k in k_keys:
            # 兼容处理：如果某个方法缺少某个K值，用0代替
            val = results[method]['top_k'].get(k, 0)
            accs.append(val * 100)

        ax.plot(k_values, accs, marker=markers[idx % len(markers)],
                linewidth=2.5, markersize=8, label=method, color=colors[idx])

        # 仅为最大K值添加文本标签，防止重叠
        ax.text(k_values[-1], accs[-1] + 1, f'{accs[-1]:.1f}%',
                ha='left', va='center', fontsize=9, color=colors[idx], fontweight='bold')

    ax.set_xlabel('Top-K', fontsize=14, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Top-K 准确率趋势对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(k_values)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {save_path}")
    plt.close()


def plot_grouped_comparison(results, save_path):
    """图3: 分组柱状图 (Top-1, Top-5, Top-10)"""
    methods = list(results.keys())

    # 尝试提取 standard keys, 假如没有则根据存在的 keys 动态调整
    target_ks = ['1', '5', '10']
    valid_ks = [k for k in target_ks if k in results[methods[0]]['top_k']]

    if not valid_ks:
        print("⚠️ 数据中缺少 Top-1/5/10 信息，跳过图3生成")
        return

    data_map = {k: [] for k in valid_ks}

    for m in methods:
        for k in valid_ks:
            data_map[k].append(results[m]['top_k'].get(k, 0) * 100)

    x = np.arange(len(methods))
    width = 0.8 / len(valid_ks)

    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制分组柱状图
    color_map = {'1': '#3498db', '5': '#e74c3c', '10': '#2ecc71', '20': '#f1c40f'}

    for i, k in enumerate(valid_ks):
        offset = (i - len(valid_ks) / 2) * width + width / 2
        bars = ax.bar(x + offset, data_map[k], width, label=f'Top-{k}',
                      color=color_map.get(k, 'gray'), alpha=0.85, edgecolor='white')

        # 数值标签
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    ax.set_title('不同方法在各 Top-K 指标下的表现', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, rotation=15, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {save_path}")
    plt.close()


def plot_improvement_analysis(results, save_path):
    """图4: 改进倍数分析"""
    random_baseline = 100 / 178  # 0.56%

    methods = ['随机猜测'] + list(results.keys())
    accuracies = [random_baseline] + [results[m]['accuracy'] * 100 for m in results.keys()]
    improvements = [1] + [(results[m]['accuracy'] * 100) / random_baseline for m in results.keys()]

    colors = ['#95a5a6'] + get_colors(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：绝对准确率
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.2,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 绝对准确率', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(methods, rotation=25, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 右图：改进倍数
    bars2 = ax2.bar(methods, improvements, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('相对随机猜测的倍数', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 相对改进倍数', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(methods, rotation=25, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('攻击效果分析：绝对准确率 vs 相对改进', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图4已保存: {save_path}")
    plt.close()


def plot_data_statistics(save_path):
    """图5: 数据集统计 (基于固定配置)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 图5.1: 节点和边数量
    categories = ['原始图', '匿名图']
    nodes = [178, 178]
    edges = [420, 315]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, nodes, width, label='节点数', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, edges, width, label='边数', color='#e74c3c', alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
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
    avg_degrees = [420 * 2 / 178, 315 * 2 / 178]

    bars = ax3.bar(graphs, avg_degrees, color=['#3498db', '#e74c3c'], alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
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
        ax4.text(width + 1, bar.get_y() + bar.get_height() / 2.,
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
    print("=" * 70)
    print("生成实验结果可视化图表 (自动识别 JSON)")
    print("=" * 70)

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 加载实验结果...")
    results = load_results()

    if not results:
        print("❌ 未加载到有效数据，程序终止。")
        return

    print(f"   检测到 {len(results)} 种实验方法: {list(results.keys())}")

    print("\n🎨 生成图表...")
    plot_accuracy_comparison(results, output_dir / 'fig1_accuracy_comparison.png')
    plot_topk_comparison(results, output_dir / 'fig2_topk_curves.png')
    plot_grouped_comparison(results, output_dir / 'fig3_grouped_comparison.png')
    plot_improvement_analysis(results, output_dir / 'fig4_improvement_analysis.png')
    plot_data_statistics(output_dir / 'fig5_data_statistics.png')

    print("\n" + "=" * 70)
    print("✅ 所有图表生成完成！")
    print(f"📁 图表保存位置: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()