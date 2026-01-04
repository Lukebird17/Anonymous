#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📊 从实验结果生成真实数据演示                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "📁 可用的实验结果文件："
    echo ""
    ls -1 results/unified/*.json 2>/dev/null | while read file; do
        echo "  • $(basename $file)"
        # 显示一些基本信息
        dataset=$(jq -r '.dataset' "$file" 2>/dev/null)
        ego_id=$(jq -r '.ego_id // "N/A"' "$file" 2>/dev/null)
        nodes=$(jq -r '.graph_stats.nodes' "$file" 2>/dev/null)
        echo "    数据集: $dataset, Ego ID: $ego_id, 节点数: $nodes"
        echo ""
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "使用方法："
    echo "  $0 <实验结果文件> [最大节点数]"
    echo ""
    echo "示例："
    echo "  $0 results/unified/facebook_ego_ego0_20251231_233954.json 50"
    echo "  $0 results/unified/cora_20251231_235254.json 30"
    echo ""
    exit 1
fi

RESULT_FILE="$1"
MAX_NODES="${2:-50}"

# 检查文件是否存在
if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ 错误: 文件不存在: $RESULT_FILE"
    exit 1
fi

# 提取数据集名称
DATASET=$(jq -r '.dataset' "$RESULT_FILE" 2>/dev/null)
EGO_ID=$(jq -r '.ego_id // "default"' "$RESULT_FILE" 2>/dev/null)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 输出文件名
OUTPUT_FILE="results/${DATASET}_${EGO_ID}_demo_${TIMESTAMP}.json"

echo "📖 输入文件: $RESULT_FILE"
echo "📊 数据集: $DATASET (Ego ID: $EGO_ID)"
echo "💾 输出文件: $OUTPUT_FILE"
echo "🎯 最大节点数: $MAX_NODES"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 运行Python脚本
python3 generate_real_demo_data.py \
    --result_file "$RESULT_FILE" \
    --output "$OUTPUT_FILE" \
    --max_nodes "$MAX_NODES"

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "✅ 成功生成演示数据！"
    echo ""
    echo "📝 下一步操作："
    echo ""
    echo "1. 更新 HTML 文件中的数据路径："
    echo "   sed -i \"s|'animated_demo_data.json'|'$(basename $OUTPUT_FILE)'|g\" results/animated_attack_demo.html"
    echo ""
    echo "2. 或者手动编辑 results/animated_attack_demo.html，将："
    echo "   fetch('animated_demo_data.json')"
    echo "   改为："
    echo "   fetch('$(basename $OUTPUT_FILE)')"
    echo ""
    echo "3. 启动演示："
    echo "   ./run_animated_demo.sh"
    echo ""
    echo "4. 浏览器访问:"
    echo "   http://localhost:8888/animated_attack_demo.html"
    echo ""
else
    echo ""
    echo "❌ 生成失败，请检查错误信息"
    exit 1
fi





