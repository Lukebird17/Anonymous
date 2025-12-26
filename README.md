# 社交网络中的结构性隐私泄露风险：基于图拓扑匹配的去匿名化分析

## 项目简介

本项目旨在证明一个核心观点：**"即便我不说话，我的朋友也会暴露我"**。

通过对社交网络进行脱敏处理（删除所有文本、头像、ID，仅保留连接关系），我们使用图拓扑特征（度分布、中介中心性、Motif模式）和图神经网络（DeepWalk、GraphSAGE）来进行去匿名化攻击实验。

## 创新点

1. **结构性去匿名化（Structural De-anonymization）**：仅基于连接模式识别用户身份
2. **对抗性实验**：模拟防御（加噪、删边）vs 攻击（图对齐、种子节点）
3. **多源数据验证**：支持微博、GitHub、Bilibili等多个社交平台数据

## 项目结构

```
deanony/
├── data/                    # 数据目录
│   ├── raw/                # 原始爬取数据
│   ├── processed/          # 处理后的图数据
│   └── anonymized/         # 匿名化后的图数据
├── crawlers/               # 爬虫模块
│   ├── weibo_crawler.py   # 微博爬虫
│   ├── github_crawler.py  # GitHub爬虫
│   └── bilibili_crawler.py # B站爬虫
├── preprocessing/          # 数据预处理
│   ├── graph_builder.py   # 构建图结构
│   ├── anonymizer.py      # 脱敏处理
│   └── perturbation.py    # 扰动添加（防御模拟）
├── models/                 # 模型实现
│   ├── deepwalk.py        # DeepWalk实现
│   ├── graphsage.py       # GraphSAGE实现
│   └── feature_extractor.py # 传统特征提取
├── attack/                 # 攻击算法
│   ├── baseline_match.py  # 基于传统特征的匹配
│   ├── embedding_match.py # 基于嵌入的匹配
│   └── graph_alignment.py # 图对齐算法
├── visualization/          # 可视化
│   ├── graph_viz.py       # 图可视化
│   └── result_viz.py      # 结果可视化
├── experiments/            # 实验脚本
│   ├── run_baseline.py    # 运行基准实验
│   ├── run_attack.py      # 运行攻击实验
│   └── run_defense.py     # 运行防御实验
├── utils/                  # 工具函数
│   ├── metrics.py         # 评估指标
│   └── config.py          # 配置文件
├── notebooks/              # Jupyter Notebooks
│   └── demo.ipynb         # 演示notebook
├── requirements.txt        # 依赖包
└── README.md              # 本文件
```

## 技术栈

- **爬虫**: requests, selenium, beautifulsoup4
- **图处理**: networkx, igraph
- **图神经网络**: PyTorch Geometric (PyG), DGL
- **图嵌入**: node2vec, DeepWalk, GraphSAGE
- **可视化**: matplotlib, seaborn, Gephi数据导出
- **数值计算**: numpy, pandas, scikit-learn

## 数据源推荐

### 1. GitHub (首选)
- **优势**: API友好，强关联性，异构图（Follow, Star, Fork, PR）
- **实验**: 抓取特定技术社区（如Rust/Go社区）的开发者关系网

### 2. 微博 (本项目重点)
- **优势**: 中文社交网络，关注/粉丝关系明确
- **挑战**: 反爬虫机制较强，需要处理动态加载

### 3. Bilibili
- **优势**: 共同关注UP主形成"社区指纹"
- **实验**: 基于用户的关注列表构建图

### 4. SNAP开放数据集 (备选)
- Facebook/Twitter/Google+ Ego networks
- 已脱敏但保留完整拓扑结构

## 实验设计

### 阶段1: 数据构建
1. 爬取原始社交网络数据
2. 构建原始图 G（完整信息）
3. 生成匿名图 G'（删除属性，保留70%边）
4. 生成已知画像图 G_base（完整拓扑）

### 阶段2: 特征提取
1. **传统方法** (Baseline)
   - 度中心性 (Degree Centrality)
   - 聚集系数 (Clustering Coefficient)
   - 介数中心性 (Betweenness Centrality)
   - k-hop邻居结构

2. **图嵌入方法**
   - DeepWalk: 随机游走 + Skip-gram
   - GraphSAGE: 归纳式图表征学习

### 阶段3: 去匿名化攻击
1. **种子节点攻击**: 假设已知5%节点作为先验
2. **图对齐**: 将匿名图嵌入空间映射到已知图空间
3. **匹配算法**: 余弦相似度 + 候选集缩减

### 阶段4: 防御对抗
1. 随机加边/删边
2. k-anonymity
3. 差分隐私扰动

### 阶段5: 评估与可视化
- **指标**: 准确率、召回率、F1-score、Top-k命中率
- **可视化**: Gephi导出、局部子图对比

## 安装与使用

### 安装依赖
```bash
cd deanony
pip install -r requirements.txt
```

### 快速开始

#### 1. 数据爬取
```bash
# 爬取微博数据
python crawlers/weibo_crawler.py --start_user <user_id> --depth 3 --max_users 5000

# 或使用GitHub数据
python crawlers/github_crawler.py --community rust --max_users 5000
```

#### 2. 数据预处理
```bash
# 构建图并脱敏
python preprocessing/graph_builder.py --input data/raw/weibo_data.json --output data/processed/
python preprocessing/anonymizer.py --input data/processed/graph.gpickle --output data/anonymized/
```

#### 3. 运行攻击实验
```bash
# 基准实验（传统特征）
python experiments/run_baseline.py --seed_ratio 0.05

# 深度学习方法
python experiments/run_attack.py --model deepwalk --seed_ratio 0.05
python experiments/run_attack.py --model graphsage --seed_ratio 0.05
```

#### 4. 可视化结果
```bash
python visualization/result_viz.py --results results/attack_results.json
```

## 实验规模建议

- **初期测试**: 5,000 - 10,000 节点
- **正式实验**: 50,000 节点左右
- **种子节点**: 5% - 10% (模拟攻击者掌握少量信息)
- **边保留率**: 70% - 90%

## 避坑指南

1. **计算量**: Graph Matching是NP-Hard问题，控制节点规模
2. **种子节点**: 纯盲匹配极难，建议5%已知节点作为引子
3. **反爬虫**: 微博/B站需要处理限流、验证码
4. **图对齐**: 两个图的Embedding空间不对齐，需要训练映射矩阵

## 预期成果

1. **论文**: 《社交网络中的结构性隐私泄露风险：基于图拓扑匹配的去匿名化分析》
2. **代码**: 完整的去匿名化攻击框架
3. **数据集**: 脱敏后的社交网络图数据
4. **可视化**: 攻击前后对比图、局部Motif展示

## 参考文献

1. Narayanan, A., & Shmatikov, V. (2009). De-anonymizing social networks. IEEE S&P.
2. Backstrom, L., et al. (2007). Wherefore art thou r3579x?: anonymized social networks, hidden patterns, and structural steganography. WWW.
3. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD.
4. Hamilton, W. L., et al. (2017). Inductive representation learning on large graphs. NIPS.

## License

MIT License

## 联系方式

如有问题，请提Issue或联系项目维护者。


