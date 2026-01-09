"""
扩展属性推断 - 同时测试Circles和Feat特征推断
对比两种推断目标的效果差异
"""

import sys
sys.path.insert(0, '.')

import os
import json
import numpy as np
import networkx as nx
from datetime import datetime
from collections import Counter

from data.dataset_loader import DatasetLoader
from data.feat_label_extractor import extract_feat_labels_from_facebook
from attack.attribute_inference import LabelPropagationAttack, AttributeInferenceAttack


def run_feat_attribute_inference(ego_id='0', hide_ratios=[0.3, 0.5, 0.7]):
    """
    运行Feat特征属性推断实验
    
    Args:
        ego_id: Ego网络ID
        hide_ratios: 隐藏比例列表
    """
    print("\n" + "="*80)
    print(f"🔬 Feat特征属性推断实验 - Ego {ego_id}")
    print("="*80)
    
    # 1. 加载数据
    loader = DatasetLoader()
    G, attributes = loader.load_facebook(ego_network=ego_id)
    
    # 2. 提取Circles标签（原有方法）
    circles_labels = {}
    for node in G.nodes():
        if node in attributes and 'circles' in attributes[node] and len(attributes[node]['circles']) > 0:
            circles_labels[node] = attributes[node]['circles'][0]  # 使用第一个circle
    
    print(f"\n📊 Circles标签统计:")
    print(f"  有标签节点: {len(circles_labels)}/{G.number_of_nodes()}")
    if circles_labels:
        label_dist = Counter(circles_labels.values())
        print(f"  唯一标签数: {len(label_dist)}")
        print(f"  前5个标签: {label_dist.most_common(5)}")
    
    # 3. 提取Feat标签（新方法）
    feat_file = f'data/datasets/facebook/{ego_id}.feat'
    featnames_file = f'data/datasets/facebook/{ego_id}.featnames'
    
    feat_labels, feat_info = extract_feat_labels_from_facebook(
        feat_file, featnames_file,
        target_category=None,  # 自动选择
        min_coverage=0.3,
        balance_threshold=0.25
    )
    
    # 4. 准备节点属性字典
    node_attributes_circles = {node: {'label': label} for node, label in circles_labels.items()}
    node_attributes_feat = {node: {'label': label} for node, label in feat_labels.items()}
    
    # 添加原始特征
    for node in G.nodes():
        if node in attributes and 'features' in attributes[node]:
            if node in node_attributes_circles:
                node_attributes_circles[node]['features'] = attributes[node]['features']
            if node in node_attributes_feat:
                node_attributes_feat[node]['features'] = attributes[node]['features']
    
    # 5. 运行两种推断实验
    results = {
        'ego_id': ego_id,
        'timestamp': datetime.now().isoformat(),
        'graph_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes()
        },
        'circles_inference': [],
        'feat_inference': [],
        'feat_info': feat_info
    }
    
    # 5.1 Circles推断
    if len(circles_labels) > 10:
        print(f"\n{'='*80}")
        print("🔵 测试1: Circles属性推断（原有方法）")
        print("="*80)
        
        for hide_ratio in hide_ratios:
            print(f"\n📍 隐藏比例: {hide_ratio:.0%}")
            print("-"*80)
            
            # 邻居投票
            lp_attacker = LabelPropagationAttack(G, node_attributes_circles)
            result_circles = lp_attacker.run_attack(train_ratio=1-hide_ratio, attribute_key='label')
            
            if 'error' not in result_circles:
                print(f"  邻居投票准确率: {result_circles['metrics']['accuracy']:.4f}")
                
                results['circles_inference'].append({
                    'hide_ratio': hide_ratio,
                    'method': 'Neighbor-Voting',
                    'accuracy': result_circles['metrics']['accuracy'],
                    'f1_macro': result_circles['metrics'].get('f1_macro', 0),
                    'n_test_samples': result_circles['metrics']['n_test_samples']
                })
    
    # 5.2 Feat推断  
    if len(feat_labels) > 10:
        print(f"\n{'='*80}")
        print("🔴 测试2: Feat特征推断（新方法 - 敏感属性）")
        print("="*80)
        print(f"  推断目标: {feat_info['category']} - {feat_info['full_name']}")
        
        for hide_ratio in hide_ratios:
            print(f"\n📍 隐藏比例: {hide_ratio:.0%}")
            print("-"*80)
            
            # 邻居投票
            lp_attacker_feat = LabelPropagationAttack(G, node_attributes_feat)
            result_feat = lp_attacker_feat.run_attack(train_ratio=1-hide_ratio, attribute_key='label')
            
            if 'error' not in result_feat:
                print(f"  邻居投票准确率: {result_feat['metrics']['accuracy']:.4f}")
                
                # 计算随机基准
                class_dist = feat_info['class_distribution']
                total = sum(class_dist.values())
                majority_baseline = max(class_dist.values()) / total
                
                print(f"  随机猜测基准: {majority_baseline:.4f}")
                print(f"  改进倍数: {result_feat['metrics']['accuracy'] / majority_baseline:.2f}x")
                
                results['feat_inference'].append({
                    'hide_ratio': hide_ratio,
                    'method': 'Neighbor-Voting',
                    'accuracy': result_feat['metrics']['accuracy'],
                    'f1_macro': result_feat['metrics'].get('f1_macro', 0),
                    'n_test_samples': result_feat['metrics']['n_test_samples'],
                    'random_baseline': majority_baseline,
                    'improvement_factor': result_feat['metrics']['accuracy'] / majority_baseline
                })
    
    # 6. 保存结果
    output_dir = 'results/feat_inference'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'ego{ego_id}_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return results


def compare_circles_vs_feat(results):
    """
    对比Circles和Feat推断的结果
    """
    print(f"\n{'='*80}")
    print("📊 Circles vs Feat 推断效果对比")
    print("="*80)
    
    if not results['circles_inference'] or not results['feat_inference']:
        print("⚠️  缺少对比数据")
        return
    
    print(f"\n🎯 推断目标对比:")
    print(f"  Circles: 社交圈标签 (学术意义 - 社区检测)")
    print(f"  Feat: {results['feat_info']['category']} (隐私风险 - 敏感属性)")
    
    print(f"\n📈 准确率对比:")
    print(f"{'隐藏比例':<10} {'Circles准确率':<15} {'Feat准确率':<15} {'差异':<10}")
    print("-"*60)
    
    for i in range(len(results['circles_inference'])):
        c = results['circles_inference'][i]
        f = results['feat_inference'][i]
        
        diff = f['accuracy'] - c['accuracy']
        diff_str = f"{diff:+.4f}"
        
        print(f"{c['hide_ratio']:<10.0%} {c['accuracy']:<15.4f} {f['accuracy']:<15.4f} {diff_str:<10}")
    
    print(f"\n💡 结论:")
    avg_circles = np.mean([r['accuracy'] for r in results['circles_inference']])
    avg_feat = np.mean([r['accuracy'] for r in results['feat_inference']])
    
    print(f"  Circles平均准确率: {avg_circles:.4f}")
    print(f"  Feat平均准确率: {avg_feat:.4f}")
    
    if avg_feat > avg_circles:
        print(f"  🔥 Feat特征推断效果更好 (+{avg_feat-avg_circles:.4f})")
        print(f"  说明: 敏感属性具有更强的同质性（如性别/学校/雇主）")
    elif avg_circles > avg_feat:
        print(f"  ℹ️  Circles推断效果更好 (+{avg_circles-avg_feat:.4f})")
        print(f"  说明: 社交圈同质性更强")
    else:
        print(f"  ⚖️  两者效果相当")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feat特征属性推断实验')
    parser.add_argument('--ego_id', type=str, default='0', help='Ego网络ID')
    parser.add_argument('--hide_ratios', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                       help='隐藏比例列表')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_feat_attribute_inference(
        ego_id=args.ego_id,
        hide_ratios=args.hide_ratios
    )
    
    # 对比结果
    compare_circles_vs_feat(results)
    
    print(f"\n{'='*80}")
    print("🎉 实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()






