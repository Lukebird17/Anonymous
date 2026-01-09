#!/bin/bash
# 快速启动演示系统

echo "🔐 图匿名化攻击与防御演示系统"
echo "================================"
echo ""
echo "选择操作："
echo "  1) 启动Web演示 (推荐)"
echo "  2) 运行实验 (Facebook Ego网络)"
echo "  3) 生成新的演示数据"
echo "  4) 查看项目说明"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🌐 启动Web服务器..."
        echo "访问: http://localhost:9000/animated_attack_demo.html"
        echo "按 Ctrl+C 停止服务器"
        echo ""
        cd results
        python3 -m http.server 9000
        ;;
    2)
        echo ""
        echo "🔬 开始运行实验..."
        python3 main_experiment_unified.py --dataset facebook_ego --ego_id 0
        echo ""
        echo "✅ 实验完成！结果保存在 results/unified/"
        ;;
    3)
        echo ""
        echo "📊 生成新的演示数据..."
        python3 generate_real_demo_data.py
        echo ""
        echo "✅ 数据生成完成！文件: results/real_data_demo.json"
        ;;
    4)
        echo ""
        cat README.md | head -50
        echo ""
        echo "查看完整文档: cat README.md 或打开 README.md"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac






