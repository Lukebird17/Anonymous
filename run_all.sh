#!/bin/bash

# 一键运行完整实验

echo "🚀 开始运行完整的去匿名化攻击实验"
echo "=================================================="
echo ""

# 步骤1: 生成示例数据
echo "📊 步骤1/4: 生成示例社交网络数据..."
python step1_generate_data.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤1失败"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# 步骤2: 构建图
echo "🔨 步骤2/4: 构建图并计算特征..."
python step2_build_graph.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤2失败"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# 步骤3: 匿名化
echo "🔒 步骤3/4: 匿名化处理..."
python step3_anonymize.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤3失败"
    exit 1
fi

echo ""
echo "=================================================="
echo ""

# 步骤4: 攻击
echo "⚔️  步骤4/4: 运行去匿名化攻击..."
python step4_attack.py
if [ $? -ne 0 ]; then
    echo "❌ 步骤4失败"
    exit 1
fi

echo ""
echo "=================================================="
echo "🎉 所有步骤完成!"
echo "=================================================="
echo ""
echo "📂 查看结果:"
echo "   cat results/attack_results.json"
echo ""


