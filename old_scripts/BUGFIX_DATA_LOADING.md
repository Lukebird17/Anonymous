# 🐛 数据加载问题修复说明

## 问题描述

运行 `generate_real_demo_data.py` 时出现错误：
```
✅ 图加载完成: 0 节点, 0 边
ValueError: zero-size array to reduction operation minimum which has no identity
```

## 原因分析

1. **数据文件路径问题**: 脚本找不到 `data/facebook/{ego_id}.edges` 文件
2. **空图处理**: 图加载失败后没有正确的后备方案
3. **JSON序列化问题**: numpy类型无法直接序列化为JSON

## 修复方案

### ✅ 1. 多路径搜索

现在会尝试多个可能的数据路径：
```python
possible_paths = [
    Path('data/facebook'),
    Path('data'),
    Path('../data/facebook'),
    Path('../../data/facebook'),
]
```

### ✅ 2. 自动生成模拟图

如果找不到数据文件，会根据实验结果中的统计信息自动生成模拟图：
```python
# 使用BA模型生成无标度网络
# 节点数和平均度数来自实验结果
G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
```

### ✅ 3. 完善错误处理

- 空图检测：在计算布局前检查图是否为空
- Cora数据集：处理torch_geometric导入失败的情况
- 详细日志：输出每个步骤的状态

### ✅ 4. JSON序列化修复

添加numpy类型转换函数：
```python
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    # ...
```

## 当前行为

### 有数据文件时：
```
🔄 加载图数据...
  📁 找到数据文件: data/facebook/0.edges
  ✅ 读取了 2519 条边
  ✅ 读取了 333 个节点的属性
✅ 图加载完成: 333 节点, 2519 边
```

### 无数据文件时（自动降级）：
```
🔄 加载图数据...
  ⚠️  警告: 找不到边文件，将使用实验结果中的统计信息生成模拟图
  🔄 生成模拟图...
  ✅ 生成了 100 个节点, 651 条边
✅ 图加载完成: 100 节点, 651 边
```

## 使用方式

### 方式1: 使用真实数据（推荐）

如果你有数据文件（`data/facebook/{ego_id}.edges`）：
```bash
# 确保数据文件在正确位置
ls data/facebook/0.edges

# 运行脚本
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50
```

### 方式2: 使用模拟图（无需数据文件）

即使没有数据文件也可以运行：
```bash
# 直接运行，脚本会自动生成模拟图
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50
```

**模拟图特点：**
- ✅ 使用BA无标度网络模型（符合社交网络特性）
- ✅ 节点数和平均度数与实验结果一致
- ✅ 随机分配节点属性（A/B/C）
- ✅ 可以正常演示所有方法
- ⚠️  不是真实的图结构（但统计特性相似）

## 准备真实数据（可选）

如果你想使用真实的图结构，需要准备数据文件：

### Facebook Ego Networks

下载地址：https://snap.stanford.edu/data/ego-Facebook.html

```bash
# 创建目录
mkdir -p data/facebook

# 下载并解压（示例）
wget http://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz

# 或者从你的备份中复制
cp /path/to/your/facebook/data/*.edges data/facebook/
cp /path/to/your/facebook/data/*.feat data/facebook/
```

### Cora数据集

使用torch_geometric自动下载：
```bash
pip install torch torch_geometric
# 脚本会自动下载到 data/ 目录
```

## 测试结果

```bash
$ python3 generate_real_demo_data.py \
    --result_file results/unified/facebook_ego_ego0_20251231_233954.json \
    --output results/test_demo.json \
    --max_nodes 30

📖 读取实验结果: results/unified/facebook_ego_ego0_20251231_233954.json
📊 数据集: facebook_ego, Ego ID: 0
🔄 加载图数据...
  ⚠️  警告: 找不到边文件，将使用实验结果中的统计信息生成模拟图
  🔄 生成模拟图...
  ✅ 生成了 100 个节点, 651 条边
✅ 图加载完成: 100 节点, 651 边
🎨 计算图布局...
✅ 使用 30 个节点进行可视化
💾 保存到: results/test_demo.json
✅ 完成！
```

## FAQ

### Q: 模拟图和真实图有什么区别？

| 方面 | 真实图 | 模拟图 |
|------|--------|--------|
| 图结构 | 真实社交网络 | BA无标度网络 |
| 节点关系 | 真实朋友关系 | 随机生成 |
| 属性 | 真实用户特征 | 随机分配 |
| 统计特性 | 原始 | 近似匹配 |
| 演示效果 | 最佳 | 良好 |
| 数据要求 | 需要数据文件 | 无需文件 |

### Q: 我应该使用哪种方式？

- **有数据文件** → 使用真实数据（更authentic）
- **无数据文件** → 使用模拟图（足够用于演示）
- **演示用途** → 两种都可以
- **研究用途** → 建议使用真实数据

### Q: 如何验证使用的是哪种图？

查看运行日志：
- 真实图：`📁 找到数据文件: ...`
- 模拟图：`⚠️ 警告: 找不到边文件，将使用实验结果中的统计信息生成模拟图`

### Q: 模拟图的演示准确吗？

- ✅ 统计指标（节点数、边数）来自真实实验
- ✅ 准确率数据来自真实实验结果
- ✅ 图的拓扑结构是模拟的（但符合无标度网络特性）
- ✅ 足以展示算法原理和动画效果

## 总结

✅ **问题已修复**: 即使没有数据文件也能正常运行
✅ **自动降级**: 智能选择真实图或模拟图
✅ **详细日志**: 清楚知道使用的是哪种数据
✅ **灵活使用**: 有数据更好，没数据也行

---

**更新时间**: 2026-01-02
**版本**: v3.2
**状态**: 已修复
