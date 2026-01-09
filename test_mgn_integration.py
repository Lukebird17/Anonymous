#!/usr/bin/env python3
"""
测试MGN整合是否成功
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_mgn_import():
    """测试MGN模块导入"""
    try:
        from models.mgn import MGNModel, MGNTrainer, build_homogeneous_data
        print("✅ MGN模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ MGN模块导入失败: {e}")
        print("   提示: 需要安装 torch_geometric")
        print("   安装命令: pip install torch-geometric")
        return False

def test_mgn_class_in_attack():
    """测试MGNAttributeInferenceAttack类"""
    try:
        from attack.graphsage_attribute_inference import MGNAttributeInferenceAttack
        print("✅ MGNAttributeInferenceAttack类导入成功")
        return True
    except ImportError as e:
        print(f"❌ MGNAttributeInferenceAttack导入失败: {e}")
        return False

def test_main_experiment_has_mgn():
    """测试main_experiment_unified.py是否包含MGN支持"""
    try:
        with open('main_experiment_unified.py', 'r') as f:
            content = f.read()
        
        has_mgn_import = 'MGNAttributeInferenceAttack' in content
        has_mgn_test = 'test_mgn' in content
        has_mgn_method = '方法4' in content and 'MGN' in content
        
        if has_mgn_import and has_mgn_test and has_mgn_method:
            print("✅ main_experiment_unified.py包含MGN支持")
            print(f"   - MGN导入: {has_mgn_import}")
            print(f"   - test_mgn参数: {has_mgn_test}")
            print(f"   - MGN测试方法: {has_mgn_method}")
            return True
        else:
            print("❌ main_experiment_unified.py缺少MGN支持")
            print(f"   - MGN导入: {has_mgn_import}")
            print(f"   - test_mgn参数: {has_mgn_test}")
            print(f"   - MGN测试方法: {has_mgn_method}")
            return False
    except Exception as e:
        print(f"❌ 检查main_experiment_unified.py失败: {e}")
        return False

def test_visualization_compatibility():
    """测试可视化代码是否兼容MGN"""
    try:
        with open('visualize_unified_auto.py', 'r') as f:
            content = f.read()
        
        # 检查是否能处理多方法
        has_method_handling = "'method'" in content and 'for item in data' in content
        
        if has_method_handling:
            print("✅ 可视化代码兼容MGN（可以处理多种方法）")
            return True
        else:
            print("⚠️  可视化代码可能需要调整")
            return False
    except Exception as e:
        print(f"❌ 检查可视化代码失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 70)
    print("MGN整合测试")
    print("=" * 70)
    
    results = []
    
    print("\n【测试1】MGN模块导入")
    results.append(test_mgn_import())
    
    print("\n【测试2】MGN攻击类导入")
    results.append(test_mgn_class_in_attack())
    
    print("\n【测试3】主实验脚本MGN支持")
    results.append(test_main_experiment_has_mgn())
    
    print("\n【测试4】可视化代码兼容性")
    results.append(test_visualization_compatibility())
    
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total} 测试")
    
    if passed == total:
        print("🎉 所有测试通过！MGN整合成功！")
        print("\n使用方法:")
        print("  python3 main_experiment_unified.py --dataset facebook_ego --ego_id 0 --mode attribute_inference --save")
        print("  # 将自动测试 Neighbor-Voting, Label-Propagation, GraphSAGE, MGN 四种方法")
    elif results[0] == False:
        print("\n⚠️  需要安装依赖:")
        print("  pip install torch torch-geometric")
        print("\n其他功能已整合完成，安装依赖后即可使用MGN")
    else:
        print("\n❌ 部分测试失败，请检查以上错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
