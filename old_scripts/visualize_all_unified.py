"""
统一可视化脚本 - 自动为所有unified JSON生成可视化
支持批量处理，自动跳过已存在的图表
"""

import json
import os
import glob
from pathlib import Path
import argparse
from visualize_unified_auto import UnifiedAutoVisualizer


def find_all_unified_jsons(unified_dir='results/unified'):
    """找到所有unified JSON文件"""
    json_files = glob.glob(os.path.join(unified_dir, '*.json'))
    return sorted(json_files)


def check_if_visualized(json_path, figures_dir='results/figures'):
    """检查JSON是否已经可视化"""
    # 从JSON文件名提取数据集名称
    json_filename = os.path.basename(json_path)
    dataset_name = json_filename.replace('.json', '')
    
    # 检查是否存在所有预期的图表
    expected_charts = [
        f'{dataset_name}_deanonymization.png',
        f'{dataset_name}_attribute_inference.png',
        f'{dataset_name}_robustness.png',
        f'{dataset_name}_comprehensive.png'
    ]
    
    # 检查防御图表（如果有防御数据）
    defense_chart = f'{dataset_name}_defense.png'
    
    # 检查基本图表是否都存在
    basic_charts_exist = all(
        os.path.exists(os.path.join(figures_dir, chart))
        for chart in expected_charts
    )
    
    return basic_charts_exist


def visualize_json(json_path, force=False, figures_dir='results/figures'):
    """可视化单个JSON文件"""
    # 检查是否已经可视化
    if not force and check_if_visualized(json_path, figures_dir):
        print(f"⏭️  跳过 (已存在): {os.path.basename(json_path)}")
        return False
    
    print(f"\n{'='*70}")
    print(f"📊 正在可视化: {os.path.basename(json_path)}")
    print(f"{'='*70}")
    
    try:
        # 创建可视化器（直接传入文件路径）
        visualizer = UnifiedAutoVisualizer(results_file=json_path)
        
        # 临时修改输出目录
        original_output_dir = visualizer.output_dir
        visualizer.output_dir = Path(figures_dir)
        visualizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成所有图表
        visualizer.generate_all_figures()
        
        # 生成报告
        visualizer.generate_text_report()
        
        # 恢复输出目录
        visualizer.output_dir = original_output_dir
        
        print(f"✅ 完成: {os.path.basename(json_path)}")
        return True
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='为所有unified JSON生成可视化'
    )
    parser.add_argument(
        '--unified-dir',
        default='results/unified',
        help='unified JSON文件目录'
    )
    parser.add_argument(
        '--figures-dir',
        default='results/figures',
        help='输出图表目录'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成所有图表（即使已存在）'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = find_all_unified_jsons(args.unified_dir)
    
    if not json_files:
        print(f"❌ 在 {args.unified_dir} 中没有找到JSON文件")
        return
    
    print(f"\n🔍 找到 {len(json_files)} 个JSON文件:")
    for json_file in json_files:
        print(f"   - {os.path.basename(json_file)}")
    
    # 统计
    total = len(json_files)
    processed = 0
    skipped = 0
    failed = 0
    
    # 处理每个JSON文件
    for json_file in json_files:
        result = visualize_json(json_file, force=args.force, figures_dir=args.figures_dir)
        
        if result:
            processed += 1
        elif result is False and not check_if_visualized(json_file, args.figures_dir):
            failed += 1
        else:
            skipped += 1
    
    # 打印总结
    print(f"\n{'='*70}")
    print(f"📈 可视化完成统计")
    print(f"{'='*70}")
    print(f"   总计: {total}")
    print(f"   ✅ 已处理: {processed}")
    print(f"   ⏭️  已跳过: {skipped}")
    print(f"   ❌ 失败: {failed}")
    print(f"\n所有图表保存在: {args.figures_dir}")


if __name__ == "__main__":
    main()


