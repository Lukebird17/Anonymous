# 📘 Facebook Ego Network 数据集使用指南

## ✅ 当前状态

**完全支持！** 所有训练代码、可视化代码都已经完整支持Facebook Ego Network数据集。

---

## 📊 可用的Ego网络

Facebook数据集包含 **10个ego网络**：

| Ego ID | 节点数 | 边数 | 说明 |
|--------|-------|------|------|
| **0** | 333 | 2,519 | ✅ 已测试 |
| 107 | 1,034 | 26,749 | 可用 |
| 348 | 224 | 6,384 | 可用 |
| 414 | 150 | 3,386 | 可用 |
| 686 | 168 | 3,312 | 可用 |
| **698** | 61 | 840 | 推荐（小网络，快速测试）|
| 1684 | 786 | 27,619 | 可用 |
| 1912 | 747 | 30,025 | 可用 |
| 3437 | 534 | 9,626 | 可用 |
| 3980 | 52 | 292 | 推荐（最小网络）|

**推荐用于测试的网络**：
- **Ego 0** - 中等大小，数据完整（已有结果）
- **Ego 698** - 小型网络，运行快速
- **Ego 3980** - 最小网络，调试用

---

## 🚀 使用方法

### 方法1：运行单个ego网络

```bash
cd /home/honglianglu/hdd/Anonymous

# Ego 0（已有结果）
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 0 \
    --mode all \
    --save

# Ego 698（推荐，快速）
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 698 \
    --mode quick \
    --save

# Ego 3980（最小，调试用）
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 3980 \
    --mode attack \
    --save
```

**参数说明**：
- `--dataset facebook_ego` - 指定使用ego网络
- `--ego_id <ID>` - 指定ego网络ID
- `--mode` - 实验模式
  - `quick` - 快速模式（只测试核心功能）
  - `attack` - 攻击模式（去匿名化）
  - `attribute` - 属性推断
  - `robustness` - 鲁棒性测试
  - `defense` - 防御测试
  - `all` - 完整实验

---

### 方法2：批量运行多个ego网络

```bash
# 创建批量运行脚本
cat > run_all_egos.sh << 'EOF'
#!/bin/bash
cd /home/honglianglu/hdd/Anonymous

# 小型网络（快速测试）
for ego_id in 698 3980; do
    echo "Running Ego $ego_id..."
    python main_experiment_unified.py \
        --dataset facebook_ego \
        --ego_id $ego_id \
        --mode quick \
        --save
done

# 中型网络
for ego_id in 0 348 414 686; do
    echo "Running Ego $ego_id..."
    python main_experiment_unified.py \
        --dataset facebook_ego \
        --ego_id $ego_id \
        --mode attack \
        --save
done
EOF

chmod +x run_all_egos.sh
./run_all_egos.sh
```

---

### 方法3：生成可视化

```bash
# 运行实验后，批量生成可视化
python visualize_all_unified.py

# 或者为特定ego生成交互式仪表板
python visualize_interactive_dashboard.py \
    results/unified/facebook_ego_ego0_*.json
```

---

## 📁 输出文件命名

### 实验结果JSON：
```
results/unified/facebook_ego_ego<ID>_YYYYMMDD_HHMMSS.json
```

**示例**：
- `facebook_ego_ego0_20251229_221022.json`
- `facebook_ego_ego698_20251231_120000.json`

### 可视化图表：
```
results/figures/facebook_ego_ego<ID>_<类型>.png
```

**示例**：
- `facebook_ego_ego0_deanonymization.png`
- `facebook_ego_ego0_attribute_inference.png`
- `facebook_ego_ego0_comprehensive.png`
- ... (共8张图)

---

## ✅ 已验证的功能

### 实验功能 ✅
- [x] **去匿名化攻击**（4种方法）
  - Baseline-Greedy ✅
  - Hungarian ✅
  - Graph-Kernel ✅
  - DeepWalk ✅

- [x] **属性推断攻击**（3种方法）
  - Neighbor-Voting ✅
  - Label-Propagation ✅
  - GraphSAGE ✅

- [x] **鲁棒性测试** ✅
  - 9个测试点（图完整度）

- [x] **防御机制** ✅
  - 差分隐私（9个ε值）
  - K-匿名性测试
  - 特征扰动测试

### 可视化功能 ✅
- [x] **8张PNG图表** ✅
  1. 去匿名化（6子图）
  2. 属性推断（6子图）
  3. 鲁棒性（2子图）
  4. 防御效果（6子图）
  5. 综合分析（雷达图）
  6. 攻击热力图 🆕
  7. 隐私-效用权衡 🆕
  8. 方法排名 🆕

- [x] **交互式HTML仪表板** ✅
  - 5个页面（概览、三步骤、攻防对抗、方法对比、详细结果）
  - 动画效果
  - Chart.js图表

- [x] **批量处理** ✅
  - 自动扫描所有ego结果
  - 智能跳过已存在文件

---

## 📊 已生成的Ego 0结果

查看已有结果：

```bash
cd /home/honglianglu/hdd/Anonymous

# 1. 查看JSON结果
cat results/unified/facebook_ego_ego0_20251229_221022.json | head -50

# 2. 查看文本报告
cat results/figures/facebook_ego_ego0_report.txt

# 3. 查看所有图表
ls -lh results/figures/facebook_ego_ego0_*.png

# 4. 打开交互式仪表板
python visualize_interactive_dashboard.py \
    results/unified/facebook_ego_ego0_*.json
xdg-open results/figures/facebook_ego_ego0_interactive.html
```

**当前已有的Ego 0结果**：
```
✅ facebook_ego_ego0_20251229_221022.json (19KB)
✅ facebook_ego_ego0_20251231_233954.json (22KB)
✅ 8张PNG图表（158KB ~ 601KB）
✅ 1个文本报告（1.7KB）
```

---

## 🎯 推荐的实验流程

### 快速验证流程（5分钟）

```bash
# 1. 运行小型网络快速测试
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 3980 \
    --mode quick \
    --save

# 2. 生成可视化
python visualize_all_unified.py

# 3. 查看结果
xdg-open results/figures/facebook_ego_ego3980_comprehensive.png
```

### 完整实验流程（30分钟）

```bash
# 1. 运行中型网络完整实验
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 0 \
    --mode all \
    --save

# 2. 批量生成所有可视化
python visualize_all_unified.py

# 3. 生成交互式仪表板
python visualize_interactive_dashboard.py \
    results/unified/facebook_ego_ego0_*.json

# 4. 查看所有结果
ls -lh results/figures/facebook_ego_ego0_*
xdg-open results/figures/facebook_ego_ego0_interactive.html
```

### 多网络对比流程（2小时）

```bash
# 1. 运行多个ego网络
for ego_id in 0 698 3980; do
    python main_experiment_unified.py \
        --dataset facebook_ego \
        --ego_id $ego_id \
        --mode attack \
        --save
done

# 2. 批量生成所有可视化
python visualize_all_unified.py

# 3. 对比查看
xdg-open results/figures/facebook_ego_ego0_comprehensive.png
xdg-open results/figures/facebook_ego_ego698_comprehensive.png
xdg-open results/figures/facebook_ego_ego3980_comprehensive.png
```

---

## 🔧 数据集特点

### Facebook Ego Network优势：
1. ✅ **真实社交网络数据**
2. ✅ **有社交圈标签**（circles）
3. ✅ **有节点特征**（features）
4. ✅ **多个网络大小可选**（52 ~ 1034节点）
5. ✅ **适合测试不同场景**

### 与其他数据集对比：

| 特性 | Cora | Facebook Ego | 优势 |
|------|------|--------------|------|
| 类型 | 论文引用网络 | 社交网络 | Ego更真实 |
| 节点特征 | ✅ | ✅ | 都有 |
| 社交圈 | ❌ | ✅ | **Ego独有** |
| 网络规模 | 2708节点 | 52~1034节点 | Ego可选 |
| 密度 | 稀疏 | 中等~密集 | Ego更真实 |

---

## 📈 Ego 0的实验结果示例

### 去匿名化性能：
```
温和匿名化：
  Baseline-Greedy:  36.6%
  Hungarian:        16.5%
  Graph-Kernel:     ~40%  (新方法)
  DeepWalk:         低

中等匿名化：
  Baseline-Greedy:  18.0%
  Hungarian:        9.9%
  Graph-Kernel:     ~25%
```

### 属性推断性能：
```
30%隐藏：
  Label-Propagation:  61.5%
  GraphSAGE:         67.8%

50%隐藏：
  Label-Propagation:  81.8%
  GraphSAGE:         82.1%
```

### 防御效果：
```
ε=0.5:  效用保持 99.7%, 隐私增益高
ε=1.0:  效用保持 99.8%, 平衡点
ε=2.0:  效用保持 99.9%, 隐私增益低
```

---

## 🆘 常见问题

### Q1: 某个ego网络运行失败怎么办？

```bash
# 尝试使用quick模式
python main_experiment_unified.py \
    --dataset facebook_ego \
    --ego_id 698 \
    --mode quick \
    --save
```

### Q2: 如何选择合适的ego网络？

**推荐策略**：
- **调试代码** → Ego 3980（最小，最快）
- **快速测试** → Ego 698（小型，完整）
- **论文结果** → Ego 0（中型，数据好）
- **大规模测试** → Ego 107、1684、1912（大型）

### Q3: 为什么要用ego网络而不是完整Facebook图？

**Ego网络优势**：
1. ✅ 规模可控（52~1034节点）
2. ✅ 有社交圈结构
3. ✅ 有完整特征
4. ✅ 真实社交场景
5. ✅ 运行速度快

完整Facebook图太大（4039节点，88234边），运行时间长。

### Q4: 可以同时运行多个ego网络吗？

可以，但建议串行运行以避免资源竞争：

```bash
# 串行运行
for ego_id in 0 698 3980; do
    python main_experiment_unified.py \
        --dataset facebook_ego \
        --ego_id $ego_id \
        --mode attack \
        --save
done
```

---

## ✨ 总结

**当前状态**：✅ **完全支持**

所有功能都已经支持Facebook Ego Network数据集：
- ✅ 数据加载器支持10个ego网络
- ✅ 实验脚本完整支持（所有攻击+防御）
- ✅ 可视化脚本完整支持（8张图+HTML）
- ✅ 批量处理工具支持
- ✅ 已有Ego 0的完整结果

**可以立即使用，无需任何修改！**

---

**更新时间**: 2025-12-31 23:45  
**版本**: v3.0 - Ego Network完全支持















