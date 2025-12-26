#!/bin/bash

# 快速开始脚本

echo "=================================================="
echo "社交网络去匿名化攻击项目 - 快速开始"
echo "=================================================="

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data/raw data/processed data/anonymized results models

# 检查Python环境
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

echo ""
echo "安装完成!"
echo ""
echo "接下来的步骤:"
echo ""
echo "1. 爬取数据:"
echo "   # GitHub数据 (推荐)"
echo "   python crawlers/github_crawler.py --token YOUR_TOKEN --language rust --max_users 1000"
echo ""
echo "   # 或使用微博数据"
echo "   python crawlers/weibo_crawler.py --cookies YOUR_COOKIES --start_uid USER_ID --max_users 1000"
echo ""
echo "2. 构建图:"
echo "   python preprocessing/graph_builder.py"
echo ""
echo "3. 匿名化:"
echo "   python preprocessing/anonymizer.py"
echo ""
echo "4. 运行攻击:"
echo "   python experiments/run_attack.py --method all --seed_ratio 0.05"
echo ""
echo "5. 查看Jupyter演示:"
echo "   cd notebooks"
echo "   jupyter notebook demo.py  # 或转换为.ipynb"
echo ""
echo "=================================================="


