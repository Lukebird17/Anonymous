# 📊 使用真实实验数据生成可视化演示

## 🎯 目标

将你已经跑完的实验结果（JSON文件）转换为可视化网页的演示数据。

## 📁 文件说明

### 输入文件
- **实验结果JSON**: `results/unified/*.json`
  - `facebook_ego_ego0_20251231_233954.json`
  - `facebook_ego_ego1912_20260101_185730.json`
  - `facebook_ego_ego3980_20260101_184139.json`
  - `cora_20251231_235254.json`

### 生成文件
- **`generate_real_demo_data.py`**: Python脚本，读取实验结果并生成演示数据
- **`generate_demo_from_results.sh`**: Shell脚本，简化使用流程

## 🚀 快速开始

### 方式一：使用Shell脚本（推荐）

```bash
# 查看可用的实验结果
./generate_demo_from_results.sh

# 使用Facebook Ego-0数据生成（50个节点）
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50

# 使用Cora数据生成（30个节点）
./generate_demo_from_results.sh results/unified/cora_20251231_235254.json 30
```

### 方式二：直接使用Python脚本

```bash
python3 generate_real_demo_data.py \
    --result_file results/unified/facebook_ego_ego0_20251231_233954.json \
    --output results/real_demo_final.json \
    --max_nodes 50
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--result_file` | 实验结果JSON文件路径 | 必需 |
| `--output` | 输出数据文件路径 | `results/real_demo_data_final.json` |
| `--max_nodes` | 最大显示节点数（太多会卡） | 50 |

## 📊 生成的数据结构

```json
{
  "meta": {
    "dataset": "facebook_ego",
    "ego_id": "0",
    "nodes": 50,
    "edges": 385,
    "timestamp": "2025-12-31T23:38:08"
  },
  "graph": {
    "nodes": [...],  // 节点坐标、属性
    "links": [...]   // 边连接
  },
  "results": {
    "deanonymization": [...],      // 去匿名化结果
    "attribute_inference": [...],  // 属性推断结果
    "defense": [...]               // 防御结果
  },
  "animations": {
    "greedy": [...],               // 贪心匹配步骤
    "hungarian": [...],            // 匈牙利算法步骤
    "graph_kernel": {...},         // 图核数据
    "deepwalk": {...},             // 随机游走
    "attribute_inference": [...],  // 属性推断步骤
    "defense": {...}               // 防御演示数据
  }
}
```

## 🔧 集成到网页

### 步骤1：生成数据

```bash
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50
```

输出：`results/facebook_ego_0_demo_20260102_120000.json`

### 步骤2：更新HTML文件

编辑 `results/animated_attack_demo.html`，找到这一行：

```javascript
fetch('animated_demo_data.json')
```

改为：

```javascript
fetch('facebook_ego_0_demo_20260102_120000.json')
```

或者使用命令自动替换：

```bash
sed -i "s|'animated_demo_data.json'|'facebook_ego_0_demo_20260102_120000.json'|g" results/animated_attack_demo.html
```

### 步骤3：启动演示

```bash
./run_animated_demo.sh
```

浏览器访问：http://localhost:8888/animated_attack_demo.html

## 📈 数据来源说明

### 阶段一：身份去匿名化

从实验结果的 `deanonymization` 字段提取：
- **贪心匹配**: 使用 `Baseline-Greedy` 的准确率
- **匈牙利算法**: 使用 `Hungarian` 的准确率
- **图核方法**: 使用 `Graph-Kernel` 的准确率
- **DeepWalk**: 使用 `DeepWalk` 的准确率

### 阶段二：属性推断

从实验结果的 `attribute_inference` 字段提取：
- **邻居投票**: 使用 `Neighbor-Voting` 的准确率
- **标签传播**: 使用 `Label-Propagation` 的准确率和迭代次数
- **GraphSAGE**: 使用 `GraphSAGE` 的准确率和F1分数

### 阶段三：防御方法

从实验结果的 `defense` 字段提取：
- **差分隐私**: 使用 `epsilon=0.1` 的边扰动数据
- **k-匿名化**: 模拟度数调整过程
- **噪声注入**: 模拟虚假节点和边的注入

### 图结构

从数据集文件中直接加载：
- **Facebook Ego**: `data/facebook/{ego_id}.edges` 和 `{ego_id}.feat`
- **Cora**: 使用 `torch_geometric.datasets.Planetoid`

## 🎨 可视化特性

### 真实数据
- ✅ 使用实际图结构（节点、边）
- ✅ 使用真实实验准确率
- ✅ 使用真实的统计数据

### 演示动画
- ✅ 根据准确率模拟成功/失败
- ✅ 基于真实图结构进行游走
- ✅ 使用实际的邻居关系推断属性

## 🔍 示例输出

```
📖 读取实验结果: results/unified/facebook_ego_ego0_20251231_233954.json
📊 数据集: facebook_ego, Ego ID: 0
📈 图统计: {'nodes': 333, 'edges': 2519, ...}
🔄 加载图数据...
✅ 图加载完成: 333 节点, 2519 边
🎨 计算图布局...
✅ 使用 50 个节点进行可视化
🔄 生成图数据...
🎬 生成动画数据...
  - 贪心匹配...
  - 匈牙利算法...
  - 图核方法...
  - DeepWalk...
  - 属性推断...
  - 防御方法...
💾 保存到: results/facebook_ego_0_demo_20260102_120000.json
✅ 完成！

📊 生成的数据统计:
  - 节点数: 50
  - 边数: 385
  - 贪心步骤: 10
  - 随机游走: 3
  - 属性推断步骤: 8
  - 去匿名化方法: 12
  - 属性推断方法: 9
  - 防御方法: 5
```

## ⚙️ 自定义选项

### 调整节点数

节点太多会导致可视化卡顿，建议：
- **小型演示**: 20-30个节点
- **中型演示**: 40-50个节点
- **大型演示**: 60-100个节点（可能较慢）

```bash
# 小型演示（快速）
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 30

# 大型演示（详细）
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 80
```

### 选择不同数据集

```bash
# Facebook Ego-0 (333个节点)
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50

# Facebook Ego-1912 (747个节点)
./generate_demo_from_results.sh results/unified/facebook_ego_ego1912_20260101_185730.json 50

# Facebook Ego-3980 (59个节点)
./generate_demo_from_results.sh results/unified/facebook_ego_ego3980_20260101_184139.json 50

# Cora (2708个节点)
./generate_demo_from_results.sh results/unified/cora_20251231_235254.json 50
```

## 🐛 故障排除

### 问题1: "文件不存在"
```bash
# 检查文件路径
ls -la results/unified/

# 使用绝对路径
./generate_demo_from_results.sh /home/honglianglu/hdd/Anonymous/results/unified/facebook_ego_ego0_20251231_233954.json
```

### 问题2: "数据集加载失败"
```bash
# 检查数据文件是否存在
ls -la data/facebook/0.edges
ls -la data/facebook/0.feat

# 如果缺失，确保运行了数据准备脚本
```

### 问题3: "网页显示空白"
```bash
# 检查数据文件是否生成
ls -la results/*demo*.json

# 检查HTML中的数据路径是否正确
grep "fetch(" results/animated_attack_demo.html
```

### 问题4: "节点太多导致卡顿"
```bash
# 减少节点数
./generate_demo_from_results.sh <结果文件> 30  # 改为30个节点
```

## 📝 注意事项

1. **节点数限制**: 脚本会自动选择度数最高的节点进行展示
2. **属性生成**: 如果数据集没有属性，会根据特征自动生成
3. **动画步骤**: 根据实际准确率模拟成功/失败
4. **布局算法**: 使用Spring Layout，每次可能略有不同

## 🎓 高级用法

### 批量生成多个数据集

```bash
for file in results/unified/*.json; do
    echo "Processing: $file"
    ./generate_demo_from_results.sh "$file" 50
done
```

### 自定义输出路径

```bash
python3 generate_real_demo_data.py \
    --result_file results/unified/facebook_ego_ego0_20251231_233954.json \
    --output custom_demo.json \
    --max_nodes 40
```

## 📚 相关文档

- `STATS_FIX.md` - 实时统计功能说明
- `ANIMATION_UPDATE_V3.md` - 动画功能更新日志
- `ANIMATION_IMPROVEMENTS.md` - 动画设计原理

---

**更新时间**: 2026-01-02
**版本**: v3.1
**状态**: 支持真实数据





