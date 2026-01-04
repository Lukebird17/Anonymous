#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔧 修复 undefined 错误                                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "✅ 已修复问题："
echo "   • graphData.edges → graphData.links"
echo "   • JSON数据中的字段名是'links'不是'edges'"
echo ""

# 重启服务器
echo "🔄 重启服务器..."
pkill -f 'python.*http.server.*8888' 2>/dev/null
sleep 1

cd results
python3 -m http.server 8888 > ../server.log 2>&1 &
SERVER_PID=$!
cd ..

sleep 2

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ 服务器已重启 (PID: $SERVER_PID)"
else
    echo "❌ 服务器启动失败"
    exit 1
fi

TIMESTAMP=$(date +%s)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 现在请访问（清除缓存）："
echo ""
echo "   http://localhost:8888/animated_attack_demo.html?v=$TIMESTAMP"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 重要提示："
echo "   1. 请按 Ctrl+Shift+R 强制刷新"
echo "   2. 或者使用无痕模式（Ctrl+Shift+N）"
echo "   3. 检查浏览器控制台（F12）应该看到："
echo "      ✅ 数据加载成功"
echo "      ✅ 左侧显示两个图"
echo ""

