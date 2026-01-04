"""
Feat特征推断可视化 - 对比Circles和Feat推断效果
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_feat_vs_circles(json_file):
    """
    可视化Circles vs Feat推断对比结果
    
    Args:
        json_file: 实验结果JSON文件路径
    """
    # 加载数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not data['circles_inference'] or not data['feat_inference']:
        print("⚠️  缺少对比数据，无法生成图表")
        return
    
    # 提取数据
    hide_ratios = [r['hide_ratio'] for r in data['circles_inference']]
    circles_acc = [r['accuracy'] for r in data['circles_inference']]
    feat_acc = [r['accuracy'] for r in data['feat_inference']]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Circles vs Feat属性推断对比 - Ego {data["ego_id"]}', 
                fontsize=16, fontweight='bold')
    
    # 图1: 准确率对比曲线
    ax1 = axes[0, 0]
    x_pos = np.array(hide_ratios) * 100
    
    ax1.plot(x_pos, circles_acc, 'o-', linewidth=2, markersize=10, 
            label='Circles (社交圈)', color='#2E86DE')
    ax1.plot(x_pos, feat_acc, 's-', linewidth=2, markersize=10,
            label=f'Feat ({data["feat_info"]["category"]})', color='#EE5A24')
    
    # 添加随机基准线（如果有）
    if 'random_baseline' in data['feat_inference'][0]:
        baseline = data['feat_inference'][0]['random_baseline']
        ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
                   label=f'随机猜测基准 ({baseline:.2%})')
    
    ax1.set_xlabel('隐藏比例 (%)', fontsize=12)
    ax1.set_ylabel('推断准确率', fontsize=12)
    ax1.set_title('(a) 准确率随隐藏比例变化', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 图2: 对比柱状图
    ax2 = axes[0, 1]
    x_pos_bar = np.arange(len(hide_ratios))
    width = 0.35
    
    bars1 = ax2.bar(x_pos_bar - width/2, circles_acc, width, 
                   label='Circles', color='#2E86DE', alpha=0.8)
    bars2 = ax2.bar(x_pos_bar + width/2, feat_acc, width,
                   label='Feat', color='#EE5A24', alpha=0.8)
    
    ax2.set_xlabel('隐藏比例', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('(b) 不同隐藏比例下的准确率对比', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos_bar)
    ax2.set_xticklabels([f'{int(r*100)}%' for r in hide_ratios])
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 图3: 改进倍数（相对于随机）
    ax3 = axes[1, 0]
    
    if 'improvement_factor' in data['feat_inference'][0]:
        feat_improvement = [r['improvement_factor'] for r in data['feat_inference']]
        
        # 计算circles的改进倍数（相对于多数类基准）
        # 假设circles也有类似的随机基准
        circles_improvement = [acc / 0.05 for acc in circles_acc]  # 假设23个类别，1/23≈0.043
        
        x_pos_bar = np.arange(len(hide_ratios))
        bars1 = ax3.bar(x_pos_bar - width/2, circles_improvement, width,
                       label='Circles', color='#2E86DE', alpha=0.8)
        bars2 = ax3.bar(x_pos_bar + width/2, feat_improvement, width,
                       label='Feat', color='#EE5A24', alpha=0.8)
        
        ax3.set_xlabel('隐藏比例', fontsize=12)
        ax3.set_ylabel('相对随机猜测的改进倍数', fontsize=12)
        ax3.set_title('(c) 相对于随机猜测的改进效果', fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos_bar)
        ax3.set_xticklabels([f'{int(r*100)}%' for r in hide_ratios])
        ax3.legend(fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='随机水平')
    
    # 图4: 特征信息和统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 准备信息文本
    info_text = f"""
📊 数据集信息:
  • Ego网络ID: {data['ego_id']}
  • 节点数: {data['graph_stats']['nodes']}
  • 边数: {data['graph_stats']['edges']}
  • 平均度: {data['graph_stats']['avg_degree']:.2f}

🎯 推断目标对比:

【Circles (社交圈)】
  • 推断内容: 节点属于哪个社交圈
  • 标签类型: 多分类 (家人/同学/同事等)
  • 敏感程度: 低 (社区检测)
  • 学术意义: 高
  • 实际威胁: 一般

【Feat (敏感属性)】
  • 推断内容: {data['feat_info']['full_name']}
  • 标签类型: 二分类
  • 特征类别: {data['feat_info']['category']}
  • 覆盖率: {data['feat_info']['coverage']:.1%}
  • 类别分布: 0={data['feat_info']['class_distribution'][0]}, 1={data['feat_info']['class_distribution'][1]}
  • 敏感程度: 高 (隐私泄露)
  • 实际威胁: 严重

📈 平均准确率:
  • Circles: {np.mean(circles_acc):.4f}
  • Feat: {np.mean(feat_acc):.4f}
  • 差异: {np.mean(feat_acc) - np.mean(circles_acc):+.4f}

💡 关键发现:
  {"Feat特征具有更强的同质性" if np.mean(feat_acc) > np.mean(circles_acc) else "Circles具有更强的同质性"}
  即使隐藏{int(max(hide_ratios)*100)}%的标签，推断准确率仍达到{min(feat_acc):.1%}
"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('results/feat_inference/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'ego{data["ego_id"]}_circles_vs_feat.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_file}")
    
    plt.show()


def generate_comparison_report(json_file):
    """
    生成文本格式的对比报告
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    output_dir = Path('results/feat_inference')
    report_file = output_dir / f'ego{data["ego_id"]}_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Circles vs Feat 属性推断对比报告 - Ego {data['ego_id']}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"生成时间: {data['timestamp']}\n\n")
        
        f.write("一、数据集信息\n")
        f.write("-"*80 + "\n")
        f.write(f"  节点数: {data['graph_stats']['nodes']}\n")
        f.write(f"  边数: {data['graph_stats']['edges']}\n")
        f.write(f"  平均度: {data['graph_stats']['avg_degree']:.2f}\n\n")
        
        f.write("二、推断目标对比\n")
        f.write("-"*80 + "\n")
        f.write("  【Circles - 社交圈标签】\n")
        f.write("    推断内容: 节点属于哪个社交圈（家人/同学/同事等）\n")
        f.write("    隐私风险: 低（社区检测，学术研究）\n\n")
        
        f.write("  【Feat - 敏感属性】\n")
        f.write(f"    推断内容: {data['feat_info']['full_name']}\n")
        f.write(f"    特征类别: {data['feat_info']['category']}\n")
        f.write(f"    覆盖率: {data['feat_info']['coverage']:.2%}\n")
        f.write(f"    类别分布: 负类={data['feat_info']['class_distribution'][0]}, "
                f"正类={data['feat_info']['class_distribution'][1]}\n")
        f.write("    隐私风险: 高（真实敏感信息泄露）\n\n")
        
        f.write("三、实验结果\n")
        f.write("-"*80 + "\n")
        f.write(f"{'隐藏比例':<12} {'Circles准确率':<18} {'Feat准确率':<18} {'差异':<12}\n")
        f.write("-"*80 + "\n")
        
        for i in range(len(data['circles_inference'])):
            c = data['circles_inference'][i]
            f_data = data['feat_inference'][i]
            diff = f_data['accuracy'] - c['accuracy']
            
            f.write(f"{c['hide_ratio']:<12.0%} {c['accuracy']:<18.4f} "
                   f"{f_data['accuracy']:<18.4f} {diff:+12.4f}\n")
        
        f.write("\n")
        
        circles_acc = [r['accuracy'] for r in data['circles_inference']]
        feat_acc = [r['accuracy'] for r in data['feat_inference']]
        
        f.write("四、统计摘要\n")
        f.write("-"*80 + "\n")
        f.write(f"  Circles平均准确率: {np.mean(circles_acc):.4f}\n")
        f.write(f"  Feat平均准确率: {np.mean(feat_acc):.4f}\n")
        f.write(f"  Circles最佳准确率: {max(circles_acc):.4f} (隐藏{data['circles_inference'][circles_acc.index(max(circles_acc))]['hide_ratio']:.0%})\n")
        f.write(f"  Feat最佳准确率: {max(feat_acc):.4f} (隐藏{data['feat_inference'][feat_acc.index(max(feat_acc))]['hide_ratio']:.0%})\n\n")
        
        f.write("五、关键发现\n")
        f.write("-"*80 + "\n")
        
        if np.mean(feat_acc) > np.mean(circles_acc):
            f.write(f"  🔥 Feat特征推断效果更好 (+{np.mean(feat_acc)-np.mean(circles_acc):.4f})\n")
            f.write("  说明: 敏感属性（如性别/学校/雇主）具有更强的同质性\n")
            f.write("  隐私风险: 即使数据被匿名化，敏感属性仍可被高准确率推断\n")
        else:
            f.write(f"  ℹ️  Circles推断效果更好 (+{np.mean(circles_acc)-np.mean(feat_acc):.4f})\n")
            f.write("  说明: 社交圈同质性强于特定敏感属性\n")
        
        f.write("\n六、隐私保护建议\n")
        f.write("-"*80 + "\n")
        f.write("  1. 仅匿名化图结构不足以保护隐私\n")
        f.write("  2. 必须同时保护或扰动节点特征\n")
        f.write("  3. 敏感属性的同质性使其容易被推断\n")
        f.write("  4. 需要考虑差分隐私等更强的保护机制\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ 报告已保存: {report_file}")


def main():
    """主函数"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Feat推断结果可视化')
    parser.add_argument('--json_file', type=str, default=None,
                       help='结果JSON文件路径（如果不指定，使用最新的）')
    
    args = parser.parse_args()
    
    # 如果未指定文件，使用最新的
    if args.json_file is None:
        json_files = glob.glob('results/feat_inference/ego*.json')
        if not json_files:
            print("❌ 未找到结果文件")
            print("请先运行: python run_feat_inference_experiment.py")
            return
        args.json_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"📁 使用最新结果文件: {args.json_file}")
    
    # 生成可视化
    visualize_feat_vs_circles(args.json_file)
    
    # 生成报告
    generate_comparison_report(args.json_file)
    
    print("\n🎉 可视化完成！")


if __name__ == "__main__":
    main()

