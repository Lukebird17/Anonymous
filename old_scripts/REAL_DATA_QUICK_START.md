# 🚀 快速开始：使用真实数据生成演示

## 一分钟快速上手

```bash
# 1. 生成演示数据（使用Facebook Ego-0，50个节点）
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50

# 2. 启动服务（会在8888端口）
./run_animated_demo.sh

# 3. 浏览器打开
http://localhost:8888/animated_attack_demo.html
```

## 🎯 核心功能

### ✅ 真实数据驱动
- 使用你跑完的实验结果
- 真实的图结构（节点、边）
- 真实的准确率和统计数据

### ✅ 自动生成
- 读取 JSON 实验结果
- 提取图结构和节点属性
- 生成适合可视化的演示数据

### ✅ 支持所有方法
- **阶段一（4个）**: 贪心、匈牙利、图核、DeepWalk
- **阶段二（3个）**: 邻居投票、标签传播、GraphSAGE
- **阶段三（3个）**: 差分隐私、k-匿名化、噪声注入

## 📁 可用数据集

| 文件 | 数据集 | Ego ID | 节点数 | 推荐显示 |
|------|--------|--------|--------|----------|
| `facebook_ego_ego0_20251231_233954.json` | Facebook | 0 | 333 | 50个 |
| `facebook_ego_ego1912_20260101_185730.json` | Facebook | 1912 | 747 | 50个 |
| `facebook_ego_ego3980_20260101_184139.json` | Facebook | 3980 | 59 | 50个 |
| `cora_20251231_235254.json` | Cora | - | 2708 | 50个 |

## 🎨 节点数选择建议

- **20-30个**: 快速演示，动画流畅
- **40-50个**: 平衡效果，推荐
- **60-80个**: 详细展示，可能稍慢
- **100+个**: 完整视图，较慢

## 📝 详细步骤

### 步骤1：生成数据

```bash
# 选择一个实验结果文件
./generate_demo_from_results.sh results/unified/facebook_ego_ego0_20251231_233954.json 50
```

输出示例：
```
📖 输入文件: results/unified/facebook_ego_ego0_20251231_233954.json
📊 数据集: facebook_ego (Ego ID: 0)
💾 输出文件: results/facebook_ego_0_demo_20260102_123456.json
🎯 最大节点数: 50

📖 读取实验结果...
🔄 加载图数据...
✅ 图加载完成: 333 节点, 2519 边
🎨 计算图布局...
✅ 使用 50 个节点进行可视化
💾 保存到: results/facebook_ego_0_demo_20260102_123456.json
✅ 完成！
```

### 步骤2：更新HTML（自动或手动）

**自动方式（推荐）：**
脚本会告诉你运行哪个命令，例如：
```bash
sed -i "s|'animated_demo_data.json'|'facebook_ego_0_demo_20260102_123456.json'|g" results/animated_attack_demo.html
```

**手动方式：**
编辑 `results/animated_attack_demo.html`，找到：
```javascript
fetch('animated_demo_data.json')
```
改为：
```javascript
fetch('facebook_ego_0_demo_20260102_123456.json')
```

### 步骤3：启动服务

```bash
./run_animated_demo.sh
```

### 步骤4：在浏览器中查看

访问：http://localhost:8888/animated_attack_demo.html

## 🎬 使用演示系统

### 选择阶段
- 阶段一：身份去匿名化
- 阶段二：属性推断攻击
- 阶段三：差分隐私防御

### 选择方法
每个阶段下有3-4个不同的方法

### 查看演示
1. 点击"开始演示"按钮
2. 观察动画演示
3. 查看实时统计（右侧）
4. 阅读原理说明（下方）

### 实时统计
- 节点数
- 当前步骤 (X/Y)
- 动态指标（匹配成功/已推断/扰动边数）
- 完成度百分比

## 🔧 故障排除

### Q1: 生成数据时出错
```bash
# 检查Python环境
python3 --version  # 需要 Python 3.6+

# 检查依赖
pip install networkx numpy
```

### Q2: 网页显示空白
```bash
# 检查数据文件
ls -la results/*demo*.json

# 检查浏览器控制台
按 F12 查看错误信息
```

### Q3: 动画太慢
```bash
# 减少节点数
./generate_demo_from_results.sh <文件> 30  # 改为30个节点
```

### Q4: 端口8888被占用
```bash
# 修改端口
python3 -m http.server 9999 -d results

# 然后访问
http://localhost:9999/animated_attack_demo.html
```

## 📊 数据说明

### 图结构
- 从 `data/facebook/{ego_id}.edges` 加载边
- 从 `data/facebook/{ego_id}.feat` 加载特征（转换为属性）
- 使用Spring Layout计算节点位置

### 准确率
- 从实验结果JSON的 `accuracy` 字段提取
- 用于模拟动画中的成功/失败

### 动画步骤
- 根据图结构生成演示步骤
- 基于准确率随机模拟结果
- 保持视觉一致性

## 💡 提示

1. **首次使用**: 建议使用默认的50个节点
2. **对比实验**: 可以生成多个数据文件并切换比较
3. **性能优化**: 节点数不要超过100个
4. **布局调整**: 可以在网页中拖拽节点调整位置

## 📚 更多信息

- 详细文档: `REAL_DATA_GUIDE.md`
- 统计功能: `STATS_FIX.md`
- 动画说明: `ANIMATION_UPDATE_V3.md`

---

**版本**: v3.1
**更新**: 2026-01-02
