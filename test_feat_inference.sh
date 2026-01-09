#!/bin/bash

# 测试Feat特征推断功能

cd /home/honglianglu/hdd/Anonymous

echo "==============================================="
echo "测试1: Feat特征提取"
echo "==============================================="
python3 data/feat_label_extractor.py

echo ""
echo "==============================================="
echo "测试2: 运行Feat属性推断实验 (Ego 0)"
echo "==============================================="
python3 run_feat_inference_experiment.py --ego_id 0

echo ""
echo "==============================================="
echo "测试3: 生成可视化"
echo "==============================================="
python3 visualize_feat_inference.py

echo ""
echo "✅ 所有测试完成！"






