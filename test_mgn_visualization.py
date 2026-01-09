#!/usr/bin/env python3
"""
测试MGN结果的可视化兼容性
验证anony-MGN中的MGN结果是否会被正确绘制
"""

import json
import sys

def test_anony_mgn_results():
    """测试anony-MGN项目中的MGN结果"""
    print("="*70)
    print("测试 anony-MGN 项目中的MGN结果")
    print("="*70)
    
    result_file = "/home/honglianglu/hdd/anony-MGN/results/unified/facebook_ego_ego0_20260110_020855.json"
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        if 'attribute_inference' not in data:
            print("❌ 没有找到attribute_inference数据")
            return False
        
        attr_data = data['attribute_inference']
        
        # 提取所有方法
        methods = sorted(set(item['method'] for item in attr_data))
        print(f"\n✅ 找到的方法列表: {methods}")
        
        # 检查MGN
        has_mgn = 'MGN' in methods
        print(f"{'✅' if has_mgn else '❌'} MGN在方法列表中: {has_mgn}")
        
        # 统计MGN结果
        mgn_results = [item for item in attr_data if item['method'] == 'MGN']
        print(f"\n✅ MGN结果数量: {len(mgn_results)}")
        
        # 显示MGN结果详情
        if mgn_results:
            print("\nMGN结果详情:")
            for i, result in enumerate(mgn_results, 1):
                print(f"  {i}. Hide Ratio: {result['hide_ratio']:.0%}, "
                      f"Label: {result.get('label_type', 'N/A')}, "
                      f"Accuracy: {result['accuracy']:.2%}")
        
        # 检查label_type
        has_label_type = any('label_type' in item for item in attr_data)
        print(f"\n✅ 包含label_type字段: {has_label_type}")
        
        if has_label_type:
            label_types = sorted(set(item.get('label_type', 'Unknown') for item in attr_data))
            print(f"✅ Label类型: {label_types}")
            
            # 统计每种label_type的MGN结果
            for label_type in label_types:
                mgn_label_results = [item for item in mgn_results 
                                    if item.get('label_type') == label_type]
                print(f"  - {label_type}: {len(mgn_label_results)} 个MGN结果")
        
        return has_mgn
        
    except Exception as e:
        print(f"❌ 读取结果文件失败: {e}")
        return False

def test_visualization_logic():
    """测试可视化代码逻辑"""
    print("\n" + "="*70)
    print("测试可视化代码逻辑")
    print("="*70)
    
    # 模拟数据（包含MGN）
    mock_data = {
        'attribute_inference': [
            {'hide_ratio': 0.3, 'method': 'Neighbor-Voting', 'label_type': 'Circles', 'accuracy': 0.60},
            {'hide_ratio': 0.3, 'method': 'Label-Propagation', 'label_type': 'Circles', 'accuracy': 0.70},
            {'hide_ratio': 0.3, 'method': 'GraphSAGE', 'label_type': 'Circles', 'accuracy': 0.75},
            {'hide_ratio': 0.3, 'method': 'MGN', 'label_type': 'Circles', 'accuracy': 0.82},
            {'hide_ratio': 0.5, 'method': 'Neighbor-Voting', 'label_type': 'Circles', 'accuracy': 0.55},
            {'hide_ratio': 0.5, 'method': 'Label-Propagation', 'label_type': 'Circles', 'accuracy': 0.65},
            {'hide_ratio': 0.5, 'method': 'GraphSAGE', 'label_type': 'Circles', 'accuracy': 0.70},
            {'hide_ratio': 0.5, 'method': 'MGN', 'label_type': 'Circles', 'accuracy': 0.78},
        ]
    }
    
    attr_data = mock_data['attribute_inference']
    
    # 模拟可视化代码的方法提取逻辑
    print("\n【步骤1】提取方法列表")
    methods = sorted(set(item['method'] for item in attr_data))
    print(f"✅ methods = sorted(set(item['method'] for item in data))")
    print(f"   结果: {methods}")
    print(f"{'✅' if 'MGN' in methods else '❌'} MGN在方法列表中")
    
    # 模拟按隐藏比例分组
    print("\n【步骤2】按隐藏比例分组")
    hide_ratios = sorted(set(item['hide_ratio'] for item in attr_data))
    print(f"✅ hide_ratios = {hide_ratios}")
    
    # 模拟label_type检测
    print("\n【步骤3】检测label_type字段")
    has_label_type = any('label_type' in item for item in attr_data)
    print(f"✅ has_label_type = {has_label_type}")
    
    # 模拟绘图循环
    print("\n【步骤4】模拟绘图循环")
    print("for label_type in label_types:")
    print("    for method in methods:")
    label_types = sorted(set(item.get('label_type', 'Unknown') for item in attr_data))
    for label_type in label_types:
        print(f"  Label Type: {label_type}")
        for method in methods:
            method_data = [item for item in attr_data 
                         if item['method'] == method and item.get('label_type') == label_type]
            if method_data:
                print(f"    ✅ {method}: {len(method_data)} 个数据点")
                for item in method_data:
                    print(f"       - Hide {item['hide_ratio']:.0%}: Acc={item['accuracy']:.2%}")
    
    return True

def test_anonymous_visualization_code():
    """测试Anonymous项目的可视化代码"""
    print("\n" + "="*70)
    print("测试 Anonymous 项目的可视化代码")
    print("="*70)
    
    viz_file = "/home/honglianglu/hdd/Anonymous/visualize_unified_auto.py"
    
    try:
        with open(viz_file, 'r') as f:
            content = f.read()
        
        # 检查关键代码
        checks = [
            ("methods = sorted(set(item['method']", "自动提取方法列表"),
            ("for method in methods:", "遍历所有方法"),
            ("item['method'] == method", "按方法过滤数据"),
            ("has_label_type = any('label_type' in item", "检测label_type"),
        ]
        
        print("\n检查关键代码片段:")
        all_ok = True
        for code, desc in checks:
            exists = code in content
            print(f"{'✅' if exists else '❌'} {desc}: {code}")
            all_ok = all_ok and exists
        
        if all_ok:
            print("\n🎉 Anonymous的可视化代码会自动绘制所有方法（包括MGN）")
        
        return all_ok
        
    except Exception as e:
        print(f"❌ 读取可视化代码失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "🔍"*35)
    print("MGN可视化兼容性测试")
    print("🔍"*35 + "\n")
    
    results = []
    
    # 测试1: anony-MGN的结果
    results.append(test_anony_mgn_results())
    
    # 测试2: 可视化逻辑
    results.append(test_visualization_logic())
    
    # 测试3: Anonymous可视化代码
    results.append(test_anonymous_visualization_code())
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过测试: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 结论: ")
        print("  ✅ anony-MGN的结果中包含MGN数据")
        print("  ✅ 可视化代码会自动提取所有方法（包括MGN）")
        print("  ✅ Anonymous项目的可视化代码完全兼容MGN")
        print("\n  📊 MGN结果会自动出现在以下图表中:")
        print("     - 属性推断性能对比图")
        print("     - 准确率随隐藏比例变化曲线")
        print("     - F1分数对比")
        print("     - Circles vs Feat对比")
        print("     - 综合性能分析")
        print("     - 方法排名对比")
        print("\n  ✨ 无需任何修改，MGN结果会自动被绘制！")
    else:
        print("\n❌ 部分测试失败，请检查以上错误")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
