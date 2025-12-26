# 使用指南

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用脚本
chmod +x quick_start.sh
./quick_start.sh
```

### 2. 数据采集

#### 方式一：爬取GitHub数据（推荐）

```python
from crawlers.github_crawler import GitHubCrawler
from pathlib import Path

# 初始化爬虫
crawler = GitHubCrawler(token="YOUR_GITHUB_TOKEN")

# 爬取Rust社区
data = crawler.crawl_network(
    language="rust",
    max_users=1000,
    max_depth=2
)

# 保存数据
crawler.save_data(data, Path("data/raw/github_data.json"))
```

**获取GitHub Token:**
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `public_repo` 权限
4. 生成并保存token

#### 方式二：爬取微博数据

```python
from crawlers.weibo_crawler import WeiboCrawler
from pathlib import Path

# 初始化爬虫（需要登录后的cookies）
crawler = WeiboCrawler(cookies="YOUR_WEIBO_COOKIES")

# 从种子用户开始爬取
data = crawler.crawl_network(
    start_uid="1234567890",  # 替换为真实用户ID
    max_users=1000,
    max_depth=2
)

# 保存数据
crawler.save_data(data, Path("data/raw/weibo_data.json"))
```

**获取微博Cookies:**
1. 登录 m.weibo.cn
2. 打开浏览器开发者工具 (F12)
3. 在Network标签中找到任意请求
4. 复制Cookie字段的值

#### 方式三：使用SNAP公开数据集

访问 https://snap.stanford.edu/data/ 下载已有的社交网络数据集，如：
- Facebook Ego Networks
- Twitter Networks
- GitHub Social Network

### 3. 数据处理

```bash
# 构建图
python preprocessing/graph_builder.py

# 匿名化处理
python preprocessing/anonymizer.py
```

或使用Python:

```python
from preprocessing.graph_builder import GraphBuilder
from preprocessing.anonymizer import GraphAnonymizer
from pathlib import Path

# 构建图
builder = GraphBuilder()
G = builder.build_from_github(Path("data/raw/github_data.json"))
G = builder.compute_node_features(G)
G = builder.extract_largest_component(G)
builder.save_graph(G, Path("data/processed/github_graph.gpickle"))

# 匿名化
anonymizer = GraphAnonymizer(edge_retention_ratio=0.7)
G_anon, node_mapping = anonymizer.anonymize(G)
ground_truth = anonymizer.create_ground_truth(G, G_anon, node_mapping)
anonymizer.save_anonymized_data(G_anon, ground_truth, Path("data/anonymized"))
```

### 4. 运行攻击实验

```bash
# 运行所有方法
python experiments/run_attack.py --method all --seed_ratio 0.05

# 只运行基准方法
python experiments/run_attack.py --method baseline

# 只运行DeepWalk方法
python experiments/run_attack.py --method deepwalk_seed --seed_ratio 0.1
```

### 5. 查看结果

```bash
# 查看结果文件
cat results/attack_report.txt

# 查看JSON结果
cat results/attack_results.json
```

## 进阶使用

### 实验1: 不同种子比例的影响

```python
import numpy as np
from experiments.run_attack import run_deepwalk_attack

seed_ratios = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
results = {}

for ratio in seed_ratios:
    result, _ = run_deepwalk_attack(G_orig, G_anon, ground_truth,
                                    use_alignment=True, seed_ratio=ratio)
    results[ratio] = result

# 可视化
from visualization.result_viz import ResultVisualizer
viz = ResultVisualizer()
viz.plot_seed_ratio_impact(results, Path("results/seed_ratio_impact.png"))
```

### 实验2: 不同边保留率的影响

```python
from preprocessing.anonymizer import GraphAnonymizer

retention_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

for ratio in retention_ratios:
    anonymizer = GraphAnonymizer(edge_retention_ratio=ratio)
    G_anon, node_mapping = anonymizer.anonymize(G)
    ground_truth = anonymizer.create_ground_truth(G, G_anon, node_mapping)
    
    result, _ = run_deepwalk_attack(G, G_anon, ground_truth,
                                    use_alignment=True, seed_ratio=0.05)
    results[ratio] = result

# 可视化
viz.plot_edge_retention_impact(results, Path("results/retention_impact.png"))
```

### 实验3: 防御机制测试

```python
# 测试k-匿名化
anonymizer = GraphAnonymizer()
G_k = anonymizer.k_anonymity(G, k=5)

# 添加噪声边
anonymizer_noise = GraphAnonymizer(
    edge_retention_ratio=0.7,
    add_noise_edges=True,
    noise_ratio=0.1
)
G_anon, mapping = anonymizer_noise.anonymize(G)
```

## Jupyter Notebook演示

```bash
cd notebooks
jupyter notebook

# 或转换demo.py为notebook
jupyter nbconvert --to notebook demo.py
```

## 常见问题

### Q1: GitHub API限流怎么办？
A: 使用Personal Access Token可获得5000次/小时的限额。如果仍不够，可以：
- 减少`max_users`参数
- 增加`delay`参数
- 使用多个token轮换

### Q2: 内存不足怎么办？
A: 对于大图（>10万节点），建议：
- 提取最大连通分量
- 减少DeepWalk的`num_walks`
- 使用稀疏矩阵存储

### Q3: 攻击准确率很低怎么办？
A: 可能原因：
- 边保留率太低（建议>0.7）
- 图太稀疏（平均度<3）
- 需要增加种子节点比例
- DeepWalk参数需要调优

### Q4: 能否用于有向图？
A: 可以。DeepWalk会自动处理有向边。对于特征提取，部分指标会区分入度和出度。

## 项目结构说明

```
deanony/
├── crawlers/          # 数据采集
├── preprocessing/     # 数据处理
├── models/           # 模型实现
├── attack/           # 攻击算法
├── visualization/    # 可视化
├── experiments/      # 实验脚本
├── utils/            # 工具函数
├── notebooks/        # 演示notebook
└── results/          # 结果输出
```

## 引用

如果这个项目对你有帮助，请引用：

```
社交网络中的结构性隐私泄露风险：基于图拓扑匹配的去匿名化分析
基于DeepWalk和GraphSAGE的社交网络去匿名化攻击框架
```


