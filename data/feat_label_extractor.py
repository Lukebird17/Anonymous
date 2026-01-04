"""
Feat特征标签提取工具 - 从Facebook .feat文件提取敏感属性
"""

import numpy as np
from typing import Dict, Tuple
from collections import Counter


def extract_feat_labels_from_facebook(feat_file: str, featnames_file: str, 
                                      target_category: str = None,
                                      min_coverage: float = 0.3,
                                      balance_threshold: float = 0.2) -> Tuple[Dict, Dict]:
    """
    从Facebook feat文件中提取特征作为标签
    
    Args:
        feat_file: .feat文件路径
        featnames_file: .featnames文件路径
        target_category: 目标特征类别（如'gender', 'education', 'work'）
                        如果为None，自动选择最佳特征
        min_coverage: 最小特征覆盖率（至少有这个比例的节点有该特征）
        balance_threshold: 类别平衡阈值（比例在[0.5-threshold, 0.5+threshold]内）
    
    Returns:
        labels: {node_id: label} 节点标签字典
        feat_info: 特征信息字典
    """
    
    # 1. 加载特征元数据
    feature_metadata = {}
    category_features = {}  # {category: [feat_ids]}
    
    with open(featnames_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            feat_id = int(parts[0])
            rest = ' '.join(parts[1:])
            category_parts = rest.split(';')
            category = category_parts[0] if category_parts else 'unknown'
            
            feature_metadata[feat_id] = {
                'category': category,
                'full_name': rest
            }
            
            if category not in category_features:
                category_features[category] = []
            category_features[category].append(feat_id)
    
    print(f"\n📊 特征类别统计:")
    for cat, feats in sorted(category_features.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(feats)} 个特征")
    
    # 2. 加载所有节点的特征向量
    node_features = {}
    with open(feat_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            features = np.array([int(x) for x in parts[1:]])
            node_features[node_id] = features
    
    n_nodes = len(node_features)
    print(f"  总节点数: {n_nodes}")
    
    # 3. 如果指定了目标类别，使用该类别的特征
    if target_category and target_category in category_features:
        candidate_features = category_features[target_category]
        print(f"\n🎯 使用指定类别: {target_category} ({len(candidate_features)} 个候选特征)")
    else:
        # 否则考虑所有特征
        candidate_features = list(feature_metadata.keys())
        print(f"\n🔍 自动选择最佳特征 ({len(candidate_features)} 个候选)")
    
    # 4. 评估每个特征的质量
    feature_scores = []
    
    for feat_id in candidate_features:
        if feat_id >= len(node_features[list(node_features.keys())[0]]):
            continue
            
        # 统计该特征的分布
        values = [node_features[nid][feat_id] for nid in node_features]
        counter = Counter(values)
        
        # 计算覆盖率（有该特征的节点比例）
        num_with_feature = counter.get(1, 0)
        coverage = num_with_feature / n_nodes
        
        # 如果覆盖率太低或太高，跳过
        if coverage < min_coverage or coverage > (1 - min_coverage):
            continue
        
        # 计算类别平衡性（越接近0.5越好）
        balance = abs(coverage - 0.5)
        
        # 如果类别严重不平衡，跳过
        if balance > balance_threshold:
            continue
        
        # 综合评分（覆盖率高 + 平衡性好）
        score = (1 - balance) * coverage
        
        feature_scores.append({
            'feat_id': feat_id,
            'category': feature_metadata[feat_id]['category'],
            'full_name': feature_metadata[feat_id]['full_name'],
            'coverage': coverage,
            'balance': balance,
            'score': score,
            'num_positive': num_with_feature,
            'num_negative': counter.get(0, 0)
        })
    
    # 5. 选择最佳特征
    if not feature_scores:
        print("⚠️  未找到符合条件的特征")
        print(f"  提示: 尝试降低min_coverage({min_coverage})或增大balance_threshold({balance_threshold})")
        return {}, {}
    
    # 按评分排序
    feature_scores.sort(key=lambda x: x['score'], reverse=True)
    best_feature = feature_scores[0]
    
    # 6. 提取标签
    labels = {}
    for node_id, features in node_features.items():
        label_value = features[best_feature['feat_id']]
        if label_value in [0, 1]:  # 只保留有效标签
            labels[node_id] = label_value
    
    # 7. 返回信息
    feat_info = {
        'feat_id': best_feature['feat_id'],
        'category': best_feature['category'],
        'full_name': best_feature['full_name'],
        'coverage': best_feature['coverage'],
        'balance': best_feature['balance'],
        'num_classes': 2,
        'class_distribution': {
            0: best_feature['num_negative'],
            1: best_feature['num_positive']
        },
        'all_candidates': feature_scores[:10]  # 保留前10个候选特征
    }
    
    print(f"\n✅ 选择的feat特征:")
    print(f"  特征ID: {feat_info['feat_id']}")
    print(f"  类别: {feat_info['category']}")
    print(f"  名称: {feat_info['full_name']}")
    print(f"  覆盖率: {feat_info['coverage']:.2%} ({best_feature['num_positive'] + best_feature['num_negative']}/{n_nodes} 节点)")
    print(f"  类别分布: 负类={best_feature['num_negative']}, 正类={best_feature['num_positive']}")
    print(f"  平衡性: {(1-best_feature['balance'])*100:.1f}% (越接近100%越好)")
    
    if len(feature_scores) > 1:
        print(f"\n📋 其他候选特征 (前5个):")
        for i, fs in enumerate(feature_scores[1:6], 1):
            print(f"  {i}. [{fs['category']}] 覆盖率={fs['coverage']:.2%}, 平衡性={(1-fs['balance'])*100:.1f}%")
    
    return labels, feat_info


def test_feat_extraction():
    """测试feat特征提取"""
    import os
    
    ego_id = '0'
    base_path = 'data/datasets/facebook'
    
    feat_file = os.path.join(base_path, f'{ego_id}.feat')
    featnames_file = os.path.join(base_path, f'{ego_id}.featnames')
    
    if not os.path.exists(feat_file):
        print(f"❌ 文件不存在: {feat_file}")
        return
    
    print("="*70)
    print(f"测试 Ego {ego_id} 的Feat特征提取")
    print("="*70)
    
    # 测试1: 自动选择最佳特征
    print("\n【测试1】自动选择最佳特征")
    print("-"*70)
    labels_auto, info_auto = extract_feat_labels_from_facebook(
        feat_file, featnames_file
    )
    print(f"\n结果: 提取到 {len(labels_auto)} 个节点的标签")
    
    # 测试2: 指定类别
    for category in ['gender', 'education', 'work', 'hometown']:
        print(f"\n【测试2】指定类别: {category}")
        print("-"*70)
        labels_cat, info_cat = extract_feat_labels_from_facebook(
            feat_file, featnames_file, target_category=category
        )
        if labels_cat:
            print(f"结果: 提取到 {len(labels_cat)} 个节点的标签")
        else:
            print(f"结果: 该类别没有符合条件的特征")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    test_feat_extraction()

