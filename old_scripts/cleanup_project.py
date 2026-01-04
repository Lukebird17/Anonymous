#!/usr/bin/env python3
"""
清理项目旧文件脚本
将不再使用的HTML、JSON和Markdown文件移动到old_scripts目录
"""

import os
import shutil
from pathlib import Path

# 项目根目录
ROOT = Path("/home/honglianglu/hdd/Anonymous")
OLD_SCRIPTS = ROOT / "old_scripts"

# 要保留的核心文件
KEEP_FILES = {
    "results/animated_attack_demo.html",  # 主演示页面
    "results/real_data_demo.json",        # 演示数据
    "results/test_highlight.html",        # 测试页面
    "README.md",                          # 项目说明
    "requirements.txt",                   # 依赖列表
    "main_experiment_unified.py",         # 实验脚本
    "generate_real_demo_data.py",         # 数据生成
}

# 要移动的旧HTML文件（results目录）
OLD_HTML = [
    "results/attack_complete.html",
    "results/attack_principles_demo.html",
    "results/attack_demo_improved.html",
    "results/attack_standalone.html",
    "results/attack_animation.html",
    "results/real_ego_enhanced_demo.html",
    "results/animated_attack_demo.html.bak",
    "results/animated_attack_demo_test.html",
    "results/stage1_complete_viz.html",
    "results/test_d3.html",
    "results/test_load.html",
]

# 要移动的旧JSON文件（results目录）
OLD_JSON = [
    "results/all_stages_demo_data.json",
    "results/animated_demo_data.json",
    "results/stage1_demo_data.json",
    "results/test_demo.json",
    "results/__demo_20260102_204544.json",
    "results/real_ego_demo_data.json",
]

# 要移动的旧Markdown文件（根目录）
OLD_MD = [
    "ANIMATION_IMPROVEMENTS.md",
    "ANIMATION_UPDATE_V3.md",
    "BUGFIX_DATA_LOADING.md",
    "DEMO_PREVIEW.md",
    "ENHANCED_DEMO_GUIDE.md",
    "IMPROVED_DEMO_GUIDE.md",
    "LIVE_SERVER_FIX.md",
    "REAL_DATA_GUIDE.md",
    "REAL_DATA_QUICK_START.md",
    "STATS_FIX.md",
    "TASK_METHOD_MAPPING.md",
    "VISUALIZATION_GUIDE.md",
    "EXPERIMENT_ANALYSIS.md",
    "FINAL_SUMMARY.md",
    "GraphSAGE使用指南.md",
    "GraphSAGE实现完成报告.md",
    "PROJECT_IMPLEMENTATION_STATUS.md",
    "UNIFIED_USAGE_GUIDE.md",
    "Ego数据集使用指南.md",
    "代码统一完成报告.md",
    "可视化改进完成报告.md",
    "实验方案完整对照.md",
    "实验逻辑详解_通俗版.md",
    "快速开始指南.md",
    "新增功能说明.md",
    "方法名自动转换说明.md",
    "结果和可视化指南.md",
    "项目全面升级完成报告.md",
    "results/COLOR_ATTRIBUTE_COMPLETE.md",
    "results/COMPLETION_SUMMARY.md",
    "results/HIGHLIGHT_FIX_README.md",
]

# 要移动的旧Shell脚本
OLD_SHELL = [
    "complete_restart.sh",
    "diagnose_issue.sh",
    "fix_browser_cache.sh",
    "generate_demo_from_results.sh",
    "quick_fix.sh",
    "archive_old_versions.sh",
]

# 要移动的旧Python脚本
OLD_PY = [
    "generate_all_stages_data.py",
    "generate_stage1_data.py",
    "visualize_all_unified.py",
]

def move_files(file_list, category):
    """移动文件列表"""
    moved = 0
    skipped = 0
    
    for file_path in file_list:
        src = ROOT / file_path
        if src.exists():
            dst = OLD_SCRIPTS / src.name
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✅ {file_path}")
                moved += 1
            except Exception as e:
                print(f"  ❌ {file_path}: {e}")
        else:
            skipped += 1
    
    return moved, skipped

def main():
    print("=" * 50)
    print("📦 开始清理项目旧文件")
    print("=" * 50)
    print()
    
    # 确保old_scripts目录存在
    OLD_SCRIPTS.mkdir(exist_ok=True)
    
    total_moved = 0
    total_skipped = 0
    
    # 移动旧HTML文件
    print("📂 清理旧HTML文件...")
    moved, skipped = move_files(OLD_HTML, "HTML")
    total_moved += moved
    total_skipped += skipped
    print()
    
    # 移动旧JSON文件
    print("📂 清理旧JSON文件...")
    moved, skipped = move_files(OLD_JSON, "JSON")
    total_moved += moved
    total_skipped += skipped
    print()
    
    # 移动旧Markdown文件
    print("📂 清理旧Markdown文件...")
    moved, skipped = move_files(OLD_MD, "MD")
    total_moved += moved
    total_skipped += skipped
    print()
    
    # 移动旧Shell脚本
    print("📂 清理旧Shell脚本...")
    moved, skipped = move_files(OLD_SHELL, "Shell")
    total_moved += moved
    total_skipped += skipped
    print()
    
    # 移动旧Python脚本
    print("📂 清理旧Python脚本...")
    moved, skipped = move_files(OLD_PY, "Python")
    total_moved += moved
    total_skipped += skipped
    print()
    
    # 移动ARCHIVE_PLAN.md
    if (ROOT / "ARCHIVE_PLAN.md").exists():
        shutil.move(str(ROOT / "ARCHIVE_PLAN.md"), str(OLD_SCRIPTS / "ARCHIVE_PLAN.md"))
        print("  ✅ ARCHIVE_PLAN.md")
        total_moved += 1
    
    print("=" * 50)
    print("✅ 清理完成！")
    print("=" * 50)
    print()
    print(f"📊 统计:")
    print(f"  移动文件: {total_moved} 个")
    print(f"  跳过文件: {total_skipped} 个 (已不存在)")
    print()
    print("📁 保留的核心文件:")
    print("  ⭐ results/animated_attack_demo.html  (主演示)")
    print("  ⭐ results/real_data_demo.json        (演示数据)")
    print("  ⭐ results/test_highlight.html        (测试页面)")
    print("  ⭐ README.md                          (项目说明)")
    print("  ⭐ main_experiment_unified.py         (实验代码)")
    print("  ⭐ generate_real_demo_data.py         (数据生成)")
    print("  ⭐ attack/ defense/ models/ ...       (核心算法)")
    print()
    print("📦 旧文件位置: old_scripts/")
    print()
    print("🚀 快速启动演示:")
    print("  cd results && python3 -m http.server 9000")
    print("  访问: http://localhost:9000/animated_attack_demo.html")
    print()

if __name__ == "__main__":
    main()

