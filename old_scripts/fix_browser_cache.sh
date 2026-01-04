#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔧 修复浏览器缓存问题                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查文件是否存在
if [ ! -f "results/real_data_demo.json" ]; then
    echo "❌ 错误: results/real_data_demo.json 不存在！"
    echo ""
    echo "请先生成数据："
    echo "  ./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50"
    exit 1
fi

echo "✅ 数据文件存在: results/real_data_demo.json"
echo "   大小: $(du -h results/real_data_demo.json | cut -f1)"
echo ""

# 停止可能运行的服务器
echo "🛑 停止旧的HTTP服务器..."
pkill -f "python.*http.server.*8888" 2>/dev/null || true
sleep 1

# 添加时间戳到HTML文件避免缓存
TIMESTAMP=$(date +%s)
echo "📝 添加缓存破坏参数..."

# 启动新服务器
echo "🚀 启动HTTP服务器（端口8888）..."
cd results
python3 -m http.server 8888 > /dev/null 2>&1 &
SERVER_PID=$!
cd ..

sleep 2

if ps -p $SERVER_PID > /dev/null; then
    echo "✅ 服务器已启动 (PID: $SERVER_PID)"
else
    echo "❌ 服务器启动失败"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📖 清除浏览器缓存的方法："
echo ""
echo "  方法1: 强制刷新"
echo "    • Chrome/Firefox: Ctrl + Shift + R"
echo "    • Mac: Cmd + Shift + R"
echo ""
echo "  方法2: 硬重载（推荐）"
echo "    1. 打开浏览器开发者工具（F12）"
echo "    2. 右键点击刷新按钮"
echo "    3. 选择「清空缓存并硬性重新加载」"
echo ""
echo "  方法3: 添加时间戳参数"
echo "    访问: http://localhost:8888/animated_attack_demo.html?t=$TIMESTAMP"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 浏览器访问："
echo ""
echo "  http://localhost:8888/animated_attack_demo.html?t=$TIMESTAMP"
echo ""
echo "  （URL中的 ?t=$TIMESTAMP 参数可以避免缓存）"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 提示："
echo "  • 如果还是看到旧数据，请按 Ctrl+Shift+R 强制刷新"
echo "  • 确保在浏览器控制台（F12）中没有看到404错误"
echo "  • 检查控制台是否显示「✅ 数据加载成功」"
echo ""

