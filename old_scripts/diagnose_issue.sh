#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔍 诊断可视化问题                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 1. 检查数据文件
echo "1️⃣  检查数据文件..."
if [ -f "results/real_data_demo.json" ]; then
    echo "   ✅ results/real_data_demo.json 存在"
    echo "   📊 文件大小: $(du -h results/real_data_demo.json | cut -f1)"
    
    # 检查JSON格式
    if python3 -c "import json; json.load(open('results/real_data_demo.json'))" 2>/dev/null; then
        echo "   ✅ JSON格式正确"
    else
        echo "   ❌ JSON格式错误！"
    fi
else
    echo "   ❌ results/real_data_demo.json 不存在！"
fi
echo ""

# 2. 检查HTML文件
echo "2️⃣  检查HTML文件..."
if [ -f "results/animated_attack_demo.html" ]; then
    echo "   ✅ results/animated_attack_demo.html 存在"
    
    # 检查加载的数据文件名
    DATA_FILE=$(grep "d3.json(" results/animated_attack_demo.html | grep -o "'[^']*\.json'" | head -1)
    echo "   📄 HTML中加载的文件: $DATA_FILE"
    
    if [[ "$DATA_FILE" == "'real_data_demo.json'" ]]; then
        echo "   ✅ 数据文件路径正确"
    else
        echo "   ⚠️  数据文件路径可能不对: $DATA_FILE"
    fi
else
    echo "   ❌ results/animated_attack_demo.html 不存在！"
fi
echo ""

# 3. 检查HTTP服务器
echo "3️⃣  检查HTTP服务器..."
if pgrep -f "python.*http.server.*8888" > /dev/null; then
    echo "   ✅ HTTP服务器正在运行（端口8888）"
    PID=$(pgrep -f "python.*http.server.*8888")
    echo "   🔧 进程ID: $PID"
else
    echo "   ❌ HTTP服务器未运行！"
    echo "   💡 请运行: ./run_animated_demo.sh"
fi
echo ""

# 4. 测试HTTP访问
echo "4️⃣  测试HTTP访问..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/real_data_demo.json | grep -q "200"; then
    echo "   ✅ 数据文件可以通过HTTP访问"
    echo "   🌐 URL: http://localhost:8888/real_data_demo.json"
else
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/real_data_demo.json)
    echo "   ❌ HTTP访问失败！状态码: $HTTP_CODE"
fi

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/animated_attack_demo.html | grep -q "200"; then
    echo "   ✅ HTML文件可以通过HTTP访问"
    echo "   🌐 URL: http://localhost:8888/animated_attack_demo.html"
else
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/animated_attack_demo.html)
    echo "   ❌ HTTP访问失败！状态码: $HTTP_CODE"
fi
echo ""

# 5. 检查浏览器可能的问题
echo "5️⃣  可能的浏览器问题..."
echo "   📋 请检查浏览器控制台（F12）："
echo ""
echo "   如果看到以下错误："
echo ""
echo "   ❌ CORS错误"
echo "      → 问题: 跨域资源共享被阻止"
echo "      → 解决: 必须通过HTTP服务器访问，不能直接打开文件"
echo ""
echo "   ❌ 404 Not Found"
echo "      → 问题: 文件路径不正确"
echo "      → 解决: 检查URL和文件位置"
echo ""
echo "   ❌ SyntaxError: JSON.parse"
echo "      → 问题: JSON文件格式错误"
echo "      → 解决: 重新生成数据文件"
echo ""
echo "   ❌ d3 is not defined"
echo "      → 问题: D3.js库未加载"
echo "      → 解决: 检查网络连接（需要CDN）"
echo ""

# 6. 生成测试URL
echo "6️⃣  测试链接..."
TIMESTAMP=$(date +%s)
echo "   🔗 带缓存破坏参数的URL:"
echo ""
echo "      http://localhost:8888/animated_attack_demo.html?t=$TIMESTAMP"
echo ""
echo "   💡 在浏览器中复制上面的URL访问"
echo ""

# 7. 提供解决方案
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔧 快速修复步骤:"
echo ""
echo "   1. 重启服务器:"
echo "      pkill -f 'python.*http.server.*8888'"
echo "      cd results && python3 -m http.server 8888 &"
echo ""
echo "   2. 清除浏览器缓存:"
echo "      按 Ctrl+Shift+R (或 Cmd+Shift+R)"
echo ""
echo "   3. 访问新URL:"
echo "      http://localhost:8888/animated_attack_demo.html?t=$TIMESTAMP"
echo ""
echo "   4. 检查控制台:"
echo "      按 F12，查看Console标签"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

