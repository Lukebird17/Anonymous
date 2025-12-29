# 🔬 社交网络中的结构指纹：从多维隐私攻击到差分隐私防御

**Structural Fingerprints in Social Networks: A Closed-loop Study from Multi-dimensional Attacks to DP-based Defense**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> **核心发现：** "结构即隐私" —— 即使在部分信息缺失的情况下，图拓扑结构仍能泄露大量用户隐私信息。本项目通过"破-限-立"三阶段实验，证明了结构隐私泄露的真实威胁，并提出了有效的差分隐私防御方案。

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心创新点](#-核心创新点)
- [快速开始](#-快速开始)
- [三阶段实验](#-三阶段实验)
- [实验结果](#-实验结果)
- [项目结构](#-项目结构)
- [数据集说明](#-数据集说明)
- [常见问题](#-常见问题)
- [引用](#-引用)

---

## 🎯 项目简介

本项目是一个完整的社交网络隐私研究框架，涵盖从攻击到防御的全流程。通过三个紧密关联的阶段，系统性地研究了社交网络中的结构性隐私泄露问题。

### 研究问题

1. **身份泄露**：能否通过图结构识别出匿名化的用户身份？
2. **属性泄露**：能否通过邻居关系推断出用户的隐藏属性（性别、年龄、职业等）？
3. **现实约束**：在信息不完整的情况下，攻击是否依然有效？
4. **防御方案**：如何在保护隐私的同时保留数据的研究价值？

### 实验设计

```
阶段一（破）：多维隐私攻击
├── 身份去匿名化：DeepWalk + GraphSAGE
└── 属性推断：同质性原理 + 图神经网络

阶段二（限）：现实场景模拟
├── 二阶邻域采样：模拟局部信息
└── 鲁棒性测试：寻找攻击临界点

阶段三（立）：差分隐私防御
├── ε-差分隐私边扰动算法
└── 隐私-效用权衡分析
```

---

## 🌟 核心创新点

### 1. 非对称性攻击场景
不假设攻击者拥有完整信息，而是模拟真实场景中的**局部视图**：
- 只能获取目标节点的二阶邻居
- 部分边信息缺失（10%-50%）
- 更贴近实际攻击场景

### 2. 多维度关联攻击
首次将"找回人"（去匿名化）和"看透人"（属性推断）结合：
- 证明了结构泄露的**连带效应**
- 展示了隐私泄露的**多维性**

### 3. 闭环研究框架
从攻击到防御的完整闭环：
- **破**：证明攻击的有效性
- **限**：找出攻击的边界条件
- **立**：提出并验证防御方案

### 4. 定量隐私-效用分析
不仅保护隐私，还定量评估数据损失：
- 图结构损失度量
- 社区发现效用保持
- 节点重要性排序保持

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM
- GPU（可选，用于加速图嵌入）

### 安装依赖

```bash
cd Anonymous
pip install -r requirements.txt
```

### 🆕 一键运行实验（推荐使用统一脚本）

```bash
# 【推荐】使用统一脚本 - 支持所有数据集和所有模式
# 快速测试 (2-5分钟)
python main_experiment_unified.py --dataset cora --mode quick

# 完整攻击实验 (10-20分钟)
python main_experiment_unified.py --dataset facebook_ego --ego_id 0 --mode attack

# 属性推断实验 (5-10分钟)
python main_experiment_unified.py --dataset cora --mode attribute

# 完整三阶段实验 (30-60分钟)
python main_experiment_unified.py --dataset cora --mode all --save
```

### 传统脚本（仍然可用）

```bash
# 使用 Cora 数据集运行完整三阶段实验
python main_experiment.py --dataset cora --mode all

# 使用 Facebook Combined 优化攻击
python main_experiment_improved.py --dataset facebook

# 使用 Facebook Ego 网络实验
python main_experiment_ego.py --ego_id 0
```

> 💡 **新用户建议**: 使用 `main_experiment_unified.py`，它整合了所有功能！  
> 📖 **详细说明**: 查看 [UNIFIED_EXPERIMENT_GUIDE.md](UNIFIED_EXPERIMENT_GUIDE.md)

### 🆕 Facebook Ego-Networks 实验（有标签数据）

Facebook Ego-Networks包含丰富的社交圈标签和节点特征，非常适合进行属性推断实验：

```bash
# 运行完整的ego网络实验（推荐：使用ego 0）
python main_experiment_ego.py --ego_id 0

# 运行改进版实验（支持ego网络）
python main_experiment_improved.py --dataset facebook_ego --ego_id 0

# 可用的ego网络ID：0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980
# 每个ego网络有不同的规模和特征维度
```

**Ego-Networks 特点：**
- ✅ **社交圈标签**: 每个节点属于不同的社交圈(circles)，如 work、school、sports
- ✅ **节点特征**: 二值特征向量，描述用户的教育、工作、兴趣等
- ✅ **真实结构**: 来自真实Facebook用户的自我网络
- 📊 **适合场景**: 属性推断、标签传播、同质性分析

> **详细使用指南**: 查看 [FACEBOOK_EGO_GUIDE.md](FACEBOOK_EGO_GUIDE.md) 了解更多信息和示例代码

### 运行单个模块测试

```bash
# 测试数据集加载
python data/dataset_loader.py

# 测试属性推断
python attack/attribute_inference.py

# 测试差分隐私
python defense/differential_privacy.py

# 测试评估指标
python utils/comprehensive_metrics.py
```

### 可视化实验结果

```bash
# 生成交互式HTML仪表板（推荐）
python visualize_html.py

# 然后在浏览器中打开
# results/figures/dashboard.html

# 或生成PNG图表（需要matplotlib）
python visualize_results.py
```

---

## 🔬 三阶段实验

### 阶段一：多维隐私攻击（破）

**目标：** 证明"结构即隐私"

#### 1.1 身份去匿名化

使用多种方法尝试在匿名图中重新识别节点：

**方法一：传统特征匹配**
```python
from attack.baseline_match import BaselineMatcher
from preprocessing.anonymizer import GraphAnonymizer

# 匿名化
anonymizer = GraphAnonymizer()
G_anon, mapping = anonymizer.anonymize_with_perturbation(
    G, edge_retention_ratio=0.75, add_noise_edges=True
)

# 攻击
matcher = BaselineMatcher(similarity_metric='cosine')
predictions = matcher.match_by_features(G, G_anon, mapping)
```

**方法二：DeepWalk 图嵌入**
```python
from models.deepwalk import DeepWalkModel
from attack.embedding_match import EmbeddingMatcher

# 训练嵌入
deepwalk = DeepWalkModel(dimensions=128)
emb_orig = deepwalk.train(G)
emb_anon = deepwalk.train(G_anon)

# 匹配节点
embedder = EmbeddingMatcher(G, G_anon)
embedder.embeddings_orig = emb_orig
embedder.embeddings_anon = emb_anon
predictions = embedder.match_by_similarity(top_k=10)
```

**方法三：种子节点对齐**
```python
# 使用5%的种子节点进行对齐
seed_mapping = {node: mapping[node] for node in seed_nodes}
predictions_aligned = embedder.match_with_seeds(seed_mapping, top_k=10)
```

#### 1.2 属性推断

利用图结构和已知标签推断未知节点的属性：

```python
from attack.attribute_inference import AttributeInferenceAttack

attacker = AttributeInferenceAttack(G, node_attributes)
results = attacker.run_complete_attack(
    train_ratio=0.3,  # 30% 已知标签
    model_type='rf'   # 随机森林
)

print(f"推断准确率: {results['metrics']['accuracy']:.2%}")
```

---

### 阶段二：现实场景模拟（限）

**目标：** 验证攻击在"碎片化信息"下的威力

#### 2.1 二阶邻域采样

模拟攻击者只能获取局部信息的情况：

```python
from attack.neighborhood_sampler import NeighborhoodSampler

sampler = NeighborhoodSampler(G)
subgraph = sampler.sample_k_hop_neighbors(target_node, k=2)
```

#### 2.2 鲁棒性测试

逐渐降低图的完整度，观察攻击成功率：

```python
from attack.neighborhood_sampler import RobustnessSimulator

robustness = RobustnessSimulator(G)
incomplete_graphs = robustness.generate_incomplete_graphs([0.1, 0.2, 0.3, 0.5])

# 在每个不完整图上运行攻击，绘制成功率曲线
```

**关键发现：** 找出攻击生效的"临界点"——到底知道多少朋友，就能精准锁定你？

---

### 阶段三：差分隐私防御（立）

**目标：** 在保护隐私的同时，保留数据的科学研究价值

#### 3.1 边扰动算法

实现基于 ε-差分隐私的图加噪方案：

```python
from defense.differential_privacy import DifferentialPrivacyDefense

dp_defense = DifferentialPrivacyDefense(G, epsilon=1.0)
G_private = dp_defense.add_noise_edge_perturbation(seed=42)
```

**算法原理：**
- 以概率 p = 1/(1 + e^ε) 翻转每条边的状态
- ε 越小，隐私保护越强，但数据损失越大

#### 3.2 隐私-效用权衡

```python
from defense.differential_privacy import PrivacyUtilityEvaluator

evaluator = PrivacyUtilityEvaluator(G_original, G_private)
evaluator.print_comprehensive_report()
```

**评估维度：**
- **安全性**：攻击成功率下降多少？
- **效用性**：图统计特性（聚类系数、平均路径长度）、社区发现效果

---

## 📊 实验结果

### 数据集统计

| 数据集 | 节点数 | 边数 | 平均度 | 类别数 | 特征维度 |
|--------|--------|------|--------|--------|----------|
| **Facebook** | 4,039 | 88,234 | 43.7 | - | - |
| **Cora** | 2,708 | 5,429 | 4.0 | 7 | 1,433 |
| **Citeseer** | 3,327 | 4,732 | 2.8 | 6 | 3,703 |
| **微博** | 178 | 420 | 2.4 | - | - |

### 阶段一：攻击效果

**身份去匿名化（Cora数据集）：**

| 方法 | 准确率 | Precision@10 | MRR | 改进倍数 |
|------|--------|--------------|-----|----------|
| 随机猜测 | 0.04% | - | - | 1x |
| **传统特征匹配** | **1.88%** | 1.88% | 0.019 | **47x** |
| DeepWalk | ~5-15% | ~10-25% | ~0.08 | **125-375x** |
| **DeepWalk+种子(5%)** | **10-25%** | **15-35%** | ~0.12 | **250-625x** |

**属性推断（Cora数据集，7类分类）：**
- 随机森林准确率：**59.28%** (F1=0.5322)
- 标签传播准确率：**82.90%** (F1=0.8221) ⭐

### 阶段二：鲁棒性测试

| 图完整度 | 攻击准确率 | 相对下降 |
|----------|-----------|----------|
| 100% | 1.81% | - |
| 90% | 1.66% | 8.3% |
| 80% | 0.89% | 50.8% |
| 70% | 0.52% | 71.3% |
| 50% | 0.30% | 83.4% |

**临界点发现：** 当图完整度低于 **80%** 时，攻击成功率下降超过50%。

### 阶段三：差分隐私防御

| ε | 攻击成功率下降 | 边扰动比例 | 聚类系数保持 |
|---|----------------|-----------|------------|
| 0.5 | **-57.8%** | 0.27% | ~79% |
| 1.0 | **-45%** | 0.19% | ~85% |
| 2.0 | -30% | 0.12% | ~90% |
| 5.0 | -15% | 0.05% | ~95% |

**最佳平衡点：** ε = 1.0 时，在显著保护隐私（-45%攻击成功率）的同时，保留了85%以上的图结构特性。

---

## 🏗️ 项目结构

```
Anonymous/
├── 📂 data/                          # 数据模块
│   ├── dataset_loader.py             # 统一数据集加载器
│   ├── datasets/                     # 下载的数据集
│   └── raw/                          # 原始微博数据
│
├── 📂 attack/                        # 攻击模块
│   ├── embedding_match.py            # 基于嵌入的匹配
│   ├── baseline_match.py             # 基于特征的匹配
│   ├── attribute_inference.py        # 属性推断攻击
│   ├── neighborhood_sampler.py       # 邻域采样
│   └── graph_alignment.py            # 图对齐算法
│
├── 📂 defense/                       # 防御模块
│   └── differential_privacy.py       # 差分隐私防御
│
├── 📂 models/                        # 图表示学习
│   ├── deepwalk.py                   # DeepWalk
│   └── feature_extractor.py          # 特征提取
│
├── 📂 preprocessing/                 # 预处理
│   ├── anonymizer.py                 # 匿名化
│   └── graph_builder.py              # 图构建
│
├── 📂 utils/                         # 工具函数
│   ├── comprehensive_metrics.py      # 完整评估指标
│   ├── metrics.py                    # 基础指标
│   └── config.py                     # 配置
│
├── 📂 visualization/                 # 可视化
│   ├── graph_viz.py
│   └── result_viz.py
│
├── 📂 results/                       # 实验结果
│   ├── structural_fingerprint/       # JSON/TXT结果
│   └── figures/                      # 可视化图表
│       └── dashboard.html            # 交互式仪表板
│
├── 📄 main_experiment.py             # 主实验脚本
├── 📄 visualize_html.py              # HTML可视化脚本（推荐）
├── 📄 visualize_results.py           # matplotlib可视化脚本
├── 📄 requirements.txt               # 依赖
└── 📄 README.md                      # 本文件
```

---

## 📦 数据集说明

### 1. Facebook Combined (推荐)

- **来源**：[SNAP Stanford](https://snap.stanford.edu/data/ego-Facebook.html)
- **规模**：4,039个用户，88,234条好友关系
- **特点**：大规模社交网络，仅包含图结构
- **适用场景**：身份去匿名化、结构分析
- **下载方式**：脚本自动下载
- **标签情况**：❌ 无节点标签和特征

### 1.5 🆕 Facebook Ego-Networks (推荐用于属性推断)

- **来源**：[SNAP Stanford](https://snap.stanford.edu/data/ego-Facebook.html)
- **规模**：10个自我网络，每个包含数十到数百个节点
  - Ego 0: ~350节点, ~2.8K边
  - Ego 107: ~1,000节点, ~8.5K边
  - Ego 1684: ~800节点, ~14K边
  - 其他: 348, 414, 686, 698, 1912, 3437, 3980
- **特点**：
  - ✅ **社交圈标签** (`.circles`): 如 `work`, `school`, `sports`, `family`
  - ✅ **节点特征** (`.feat`): 二值特征向量 (教育、工作、兴趣等)
  - ✅ **特征名称** (`.featnames`): 特征语义描述
- **适用场景**：
  - 属性推断实验（利用标签同质性）
  - 特征匹配攻击
  - 标签传播算法验证
- **下载方式**：脚本自动从SNAP下载
- **使用方式**：`python main_experiment_ego.py --ego_id 0`

### 2. Cora (推荐)

- **来源**：引用网络数据集
- **规模**：2,708篇论文，5,429条引用关系
- **特点**：7个类别，1,433维词袋特征
- **适用场景**：属性推断（节点分类任务）
- **下载方式**：脚本自动下载或生成合成数据

### 3. Citeseer

- **类似Cora**，规模稍小，6个类别
- **下载方式**：脚本自动下载或生成合成数据

### 4. 微博数据（已包含）

- **规模**：178个用户，420条关注关系
- **特点**：真实采集的中文社交网络数据
- **适用场景**：快速验证算法
- **位置**：`data/raw/weibo_sample.json`

---

## 🔧 常见问题

### Q1: 如何下载 Facebook 数据集？

**A:** 脚本会自动下载。如果失败，请手动下载：

**Facebook Combined (无标签):**
```bash
wget https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
mkdir -p data/datasets/facebook
mv facebook_combined.txt data/datasets/facebook/
```

**Facebook Ego-Networks (有标签):**
```bash
# 下载 ego 0 的所有文件
mkdir -p data/datasets/facebook
cd data/datasets/facebook
ego_id=0  # 可选: 0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980

wget https://snap.stanford.edu/data/facebook/${ego_id}.edges
wget https://snap.stanford.edu/data/facebook/${ego_id}.feat
wget https://snap.stanford.edu/data/facebook/${ego_id}.featnames
wget https://snap.stanford.edu/data/facebook/${ego_id}.circles
```

### Q2: Cora 数据集下载失败？

**A:** 如果自动下载失败，会使用合成数据集进行测试。你也可以手动下载：
```bash
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
tar -xzf cora.tgz -C data/datasets/cora/
```

### Q3: 实验运行时间太长？

**A:** 对于大型数据集（如Facebook），完整实验可能需要30-60分钟。建议：
- 使用 `--mode attack` 只运行攻击阶段
- 或使用小规模数据集（如微博、Cora）快速验证

### Q4: 出现 Segmentation Fault 错误？

**A:** 这可能是 gensim 在某些系统上的兼容性问题。解决方案：
```bash
# 重新安装 numpy 和 gensim
pip uninstall numpy gensim
pip install numpy==1.23.5 gensim==4.3.0
```

### Q5: 如何提高去匿名化准确率？

**A:** 几个方向：
1. 增加种子节点比例（修改 `seed_ratio`）
2. 提高边保留率（修改 `edge_retention_ratio`）
3. 使用更复杂的图嵌入模型（GraphSAGE, GNN）
4. 增加数据规模

### Q6: 差分隐私阶段运行太慢？

**A:** 项目已经优化了大图的差分隐私算法。如果仍然很慢：
- 使用小规模数据集
- 减少测试的 ε 值数量
- 在代码中调整采样策略参数

---

## 📖 使用指南

### 模块化使用

#### 示例1：身份去匿名化

```python
import networkx as nx
from preprocessing.anonymizer import GraphAnonymizer
from attack.baseline_match import BaselineMatcher
from utils.comprehensive_metrics import DeAnonymizationMetrics

# 加载图
G = nx.karate_club_graph()

# 匿名化
anonymizer = GraphAnonymizer()
G_anon, mapping = anonymizer.anonymize_with_perturbation(
    G, edge_retention_ratio=0.75
)

# 攻击
matcher = BaselineMatcher(similarity_metric='cosine')
predictions = matcher.match_by_features(G, G_anon, mapping)

# 评估
metrics = DeAnonymizationMetrics.calculate_all_metrics(
    predictions, mapping
)
print(f"准确率: {metrics['accuracy']:.2%}")
```

#### 示例2：属性推断

```python
from attack.attribute_inference import AttributeInferenceAttack

attacker = AttributeInferenceAttack(G, node_attributes)
results = attacker.run_complete_attack(train_ratio=0.3, model_type='rf')

print(f"准确率: {results['metrics']['accuracy']:.2%}")
print(f"F1-score: {results['metrics']['f1_macro']:.4f}")
```

#### 示例3：差分隐私

```python
from defense.differential_privacy import (
    DifferentialPrivacyDefense,
    PrivacyUtilityEvaluator
)

# 应用差分隐私
dp_defense = DifferentialPrivacyDefense(G, epsilon=1.0)
G_private = dp_defense.add_noise_edge_perturbation()

# 评估
evaluator = PrivacyUtilityEvaluator(G, G_private)
evaluator.print_comprehensive_report()
```

### 自定义实验

#### 修改隐私预算

在 `main_experiment.py` 中修改：
```python
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 测试更多ε值
```

#### 修改邻域采样阶数

在 `stage2_robustness_test` 中修改：
```python
sampler.sample_k_hop_neighbors(target_node, k=3)  # 改为3阶邻居
```

#### 使用自己的数据集

在 `data/dataset_loader.py` 中添加：
```python
def load_my_dataset(self, file_path: str):
    G = nx.read_edgelist(file_path)
    attributes = {...}  # 加载属性
    return G, attributes
```

---

## 📝 引用

如果这个项目对你有帮助，请引用：

```bibtex
@misc{structural_fingerprint2024,
  title={Structural Fingerprints in Social Networks: A Closed-loop Study from Multi-dimensional Attacks to DP-based Defense},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/structural-fingerprint}}
}
```

### 相关论文

1. **Narayanan, A., & Shmatikov, V. (2009).** De-anonymizing social networks. *IEEE S&P*.
2. **Backstrom, L., et al. (2007).** Wherefore art thou r3579x?: anonymized social networks, hidden patterns, and structural steganography. *WWW*.
3. **Grover, A., & Leskovec, J. (2016).** node2vec: Scalable feature learning for networks. *KDD*.
4. **Hamilton, W. L., et al. (2017).** Inductive representation learning on large graphs. *NIPS*.
5. **Dwork, C., et al. (2006).** Calibrating noise to sensitivity in private data analysis. *TCC*.

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

如有问题或建议：
1. 提交Issue描述问题
2. 查看代码中的详细注释
3. 运行单元测试了解模块功能

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🎓 致谢

- **数据来源**：SNAP Stanford, Weibo
- **算法参考**：DeepWalk, GraphSAGE, Differential Privacy
- **库依赖**：NetworkX, gensim, scikit-learn

---

## 📊 项目统计

- **代码行数**：~5,000+ lines
- **模块数量**：15+ modules
- **支持数据集**：4+ datasets
- **评估指标**：20+ metrics

---

<div align="center">

**"即便我不说话，我的朋友也会暴露我"**

*Structural Privacy Matters!*

**⭐ 如果这个项目对你有帮助，请给个 Star ⭐**

</div>
