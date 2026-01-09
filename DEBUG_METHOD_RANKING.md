# Method Ranking 图表 Debug 报告

## 问题描述
用户反馈：method_ranking 图的左边子图没有显示

## 问题根源
原始数据文件 `facebook_ego_ego0_20260110_020855.json` **只包含属性推断数据，缺少去匿名化数据**：

- ✅ attribute_inference: 24 项（含 MGN）
- ❌ deanonymization: 0 项 **← 导致左图为空**
- ❌ robustness: 0 项
- ❌ defense: 0 项

左图需要 `results['deanonymization']` 数据来绘制去匿名化方法排名，但该数据不存在。

## 解决方案
**数据合并策略**：将完整数据中的去匿名化结果合并到 MGN 数据中

### 步骤 1: 合并数据源
```python
# 源1: MGN数据（facebook_ego_ego0_20260110_020855.json）
# - 包含最新的 MGN 属性推断结果

# 源2: 完整数据（facebook_ego_ego0_20260104_190950.json）
# - 包含去匿名化、鲁棒性、防御数据

# 合并输出: facebook_ego_ego0_20260110_020855_merged.json
```

### 步骤 2: 合并后数据内容
```
✅ deanonymization: 12 项  ← 从完整数据添加
✅ attribute_inference: 24 项 ← 保留 MGN 数据
✅ robustness: 9 项  ← 从完整数据添加
✅ defense: 9 项  ← 从完整数据添加
✨ MGN结果: 6 项
```

### 步骤 3: 重新生成图表
```bash
python3 visualize_unified_auto.py results/unified/facebook_ego_ego0_20260110_020855_merged.json
```

## 验证结果

### 图片质量验证
- ✅ 尺寸: 4762 x 1764 像素
- ✅ 格式: PNG (RGBA)
- ✅ 文件大小: 181.6 KB

### 内容验证
- ✅ **左半部分**: 35.4% 非白色像素（有内容）
- ✅ **右半部分**: 46.9% 非白色像素（有内容）

### 子图内容
**左图 - 去匿名化方法排名**
- 数据源: deanonymization (12 项)
- 显示内容: 各方法综合得分排名（横向条形图）
- 排名算法: accuracy×0.5 + precision@5×0.3 + MRR×0.2

**右图 - 属性推断方法排名**
- 数据源: attribute_inference (24 项)
- 显示内容: 各方法平均性能（含 MGN）
- ✨ **包含 MGN 方法**

## 最终状态
🎉 **问题完全解决**

- ✅ 左右两个子图都正常显示
- ✅ MGN 结果完整呈现在右图中
- ✅ 所有 8 张图表全部生成
- ✅ 数据完整性验证通过

## 文件位置
- 合并后数据: `results/unified/facebook_ego_ego0_20260110_020855_merged.json`
- 生成图表: `results/figures/facebook_ego_ego0_method_ranking.png`
- 所有图表: `results/figures/facebook_ego_ego0_*.png`

## 技术要点
1. **数据完整性**: 可视化脚本需要完整的实验数据
2. **数据合并**: 保持 MGN 最新结果的同时补充其他阶段数据
3. **向后兼容**: 合并策略不影响原有数据结构

---
生成时间: 2026-01-10 04:30
状态: ✅ 已解决
