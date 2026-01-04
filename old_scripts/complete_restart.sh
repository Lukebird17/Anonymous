#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔄 完全重启并清除缓存                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. 停止所有旧服务器
echo "1️⃣  停止旧服务器..."
pkill -f 'python.*http.server.*8888' 2>/dev/null && echo "   ✅ 已停止" || echo "   ℹ️  没有运行的服务器"
sleep 2

# 2. 清理旧文件（可选）
echo ""
echo "2️⃣  检查文件..."
if [ -f "results/real_data_demo.json" ]; then
    echo "   ✅ 数据文件存在 ($(du -h results/real_data_demo.json | cut -f1))"
else
    echo "   ⚠️  数据文件不存在，正在生成..."
    ./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50
fi

# 3. 添加缓存破坏版本号到HTML
echo ""
echo "3️⃣  更新HTML避免缓存..."
VERSION=$(date +%s)

# 备份原文件
cp results/animated_attack_demo.html results/animated_attack_demo.html.bak

# 在JSON加载处添加版本号
sed -i "s|d3.json('real_data_demo.json')|d3.json('real_data_demo.json?v=$VERSION')|g" results/animated_attack_demo.html

echo "   ✅ 已添加版本号参数: ?v=$VERSION"

# 4. 启动新服务器
echo ""
echo "4️⃣  启动新服务器..."
cd results
python3 -m http.server 8888 > ../server.log 2>&1 &
SERVER_PID=$!
cd ..

sleep 2

if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "   ✅ 服务器已启动 (PID: $SERVER_PID)"
else
    echo "   ❌ 服务器启动失败"
    cat server.log
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ 完成！现在请在浏览器中："
echo ""
echo "   1. 按 Ctrl+Shift+R 强制刷新"
echo "   2. 或者使用无痕模式（Ctrl+Shift+N）"
echo "   3. 访问："
echo ""
echo "      http://localhost:8888/animated_attack_demo.html?t=$VERSION"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 提示："
echo "   • 如果还是不行，请完全关闭浏览器后重新打开"
echo "   • 服务器日志：tail -f server.log"
echo "   • 恢复原HTML：mv results/animated_attack_demo.html.bak results/animated_attack_demo.html"
echo ""

