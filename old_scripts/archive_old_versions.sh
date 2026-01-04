#!/bin/bash
# 自动归档旧版本文件

cd /home/honglianglu/hdd/Anonymous

echo "========================================="
echo "📦 开始归档旧版本文件"
echo "========================================="
echo ""

# 创建归档目录
mkdir -p OLD_VERSIONS/{scripts,html,data,docs,shell_scripts}

# 计数器
moved=0
skipped=0

# 归档旧的可视化脚本
echo "📂 归档旧的可视化脚本..."
for file in visualize_html.py visualize_all_unified.py visualize_attack_principles.py visualize_attack_principles_v2.py visualize_complete_template.py visualize_principles_complete.py visualize_unified_auto.py generate_all_stages_data.py generate_stage1_data.py; do
    if [ -f "$file" ]; then
        mv "$file" OLD_VERSIONS/scripts/
        echo "  ✅ $file"
        ((moved++))
    else
        ((skipped++))
    fi
done

# 归档旧的HTML
echo ""
echo "📂 归档旧的HTML页面..."
cd results
for file in attack_complete.html attack_principles_demo.html attack_demo_improved.html attack_standalone.html attack_animation.html real_ego_enhanced_demo.html animated_attack_demo.html.bak animated_attack_demo_test.html stage1_complete_viz.html test_d3.html test_load.html; do
    if [ -f "$file" ]; then
        mv "$file" ../OLD_VERSIONS/html/
        echo "  ✅ $file"
        ((moved++))
    else
        ((skipped++))
    fi
done
cd ..

# 归档旧的数据文件
echo ""
echo "📂 归档旧的数据文件..."
cd results
for file in all_stages_demo_data.json animated_demo_data.json stage1_demo_data.json test_demo.json __demo_20260102_204544.json real_ego_demo_data.json; do
    if [ -f "$file" ]; then
        mv "$file" ../OLD_VERSIONS/data/
        echo "  ✅ $file"
        ((moved++))
    else
        ((skipped++))
    fi
done
cd ..

# 归档旧的shell脚本
echo ""
echo "📂 归档旧的shell脚本..."
for file in complete_restart.sh diagnose_issue.sh fix_browser_cache.sh generate_demo_from_results.sh quick_fix.sh run_demo.sh run_demo_v2.sh run_complete_demo.sh run_complete_viz.sh run_real_ego_demo.sh run_animated_demo.sh; do
    if [ -f "$file" ]; then
        mv "$file" OLD_VERSIONS/shell_scripts/
        echo "  ✅ $file"
        ((moved++))
    else
        ((skipped++))
    fi
done

# 归档旧的文档
echo ""
echo "📂 归档旧的文档..."
for pattern in "ANIMATION_*.md" "BUGFIX_*.md" "DEMO_*.md" "ENHANCED_*.md" "IMPROVED_*.md" "LIVE_*.md" "REAL_DATA_*.md" "STATS_*.md" "TASK_*.md" "VISUALIZATION_*.md"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            mv "$file" OLD_VERSIONS/docs/
            echo "  ✅ $file"
            ((moved++))
        fi
    done
done

# 归档results里的旧文档
cd results
for file in COLOR_*.md COMPLETION_*.md HIGHLIGHT_*.md; do
    if [ -f "$file" ]; then
        mv "$file" ../OLD_VERSIONS/docs/
        echo "  ✅ $file"
        ((moved++))
    fi
done
cd ..

# 归档FINAL_PROJECT（如果不需要）
if [ -d "FINAL_PROJECT" ]; then
    echo ""
    echo "📂 归档FINAL_PROJECT..."
    mv FINAL_PROJECT OLD_VERSIONS/
    echo "  ✅ FINAL_PROJECT 目录"
    ((moved++))
fi

echo ""
echo "========================================="
echo "✅ 归档完成！"
echo "========================================="
echo ""
echo "📊 统计："
echo "  移动文件: $moved 个"
echo "  跳过文件: $skipped 个 (已不存在)"
echo ""
echo "📁 保留的核心文件："
echo "  ⭐ results/animated_attack_demo.html"
echo "  ⭐ results/real_data_demo.json"
echo "  ⭐ results/test_highlight.html"
echo "  ⭐ main_experiment_unified.py"
echo "  ⭐ generate_real_demo_data.py"
echo "  ⭐ visualize_interactive_dashboard.py"
echo "  ⭐ attack/ defense/ models/ 等核心目录"
echo ""
echo "📦 归档位置: OLD_VERSIONS/"
echo ""
echo "🚀 快速启动演示："
echo "  cd results && python3 -m http.server 9000"
echo "  访问: http://localhost:9000/animated_attack_demo.html"
echo ""

