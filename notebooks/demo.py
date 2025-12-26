"""
演示Notebook：完整的去匿名化攻击流程
"""

# 这是一个Jupyter Notebook的Python版本
# 可以使用 jupyter nbconvert --to notebook demo.py 转换为.ipynb

# # 社交网络去匿名化攻击演示

# ## 1. 导入库

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent))

from preprocessing.graph_builder import GraphBuilder
from preprocessing.anonymizer import GraphAnonymizer
from models.deepwalk import DeepWalk
from attack.embedding_match import EmbeddingMatcher
from attack.graph_alignment import GraphAligner
from visualization.graph_viz import GraphVisualizer
from utils.metrics import calculate_accuracy, calculate_top_k_accuracy

plt.style.use('seaborn-v0_8')

# ## 2. 数据准备

# ### 2.1 使用示例数据（如果没有真实数据）

import networkx as nx

# 生成一个示例社交网络
G = nx.barabasi_albert_graph(500, 5)
print(f"示例图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 计算节点特征
builder = GraphBuilder()
G = builder.compute_node_features(G)

# 保存示例图
output_dir = Path("../data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
builder.save_graph(G, output_dir / "example_graph.gpickle")

# ## 3. 匿名化

anonymizer = GraphAnonymizer(
    edge_retention_ratio=0.7,
    add_noise_edges=True,
    noise_ratio=0.05
)

G_anon, node_mapping = anonymizer.anonymize(G)
ground_truth = anonymizer.create_ground_truth(G, G_anon, node_mapping)

print(f"匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
print(f"边保留率: {ground_truth['edge_retention_ratio']:.2%}")

# 保存匿名化数据
anon_dir = Path("../data/anonymized")
anonymizer.save_anonymized_data(G_anon, ground_truth, anon_dir)

# ## 4. 可视化对比

visualizer = GraphVisualizer(figsize=(15, 6))
visualizer.plot_comparison(G, G_anon, Path("../results/graph_comparison.png"))

# ## 5. 特征提取 - DeepWalk

# 训练原始图
model_orig = DeepWalk(dimensions=64, walk_length=40, num_walks=10)
model_orig.fit(G)

# 训练匿名图
model_anon = DeepWalk(dimensions=64, walk_length=40, num_walks=10)
model_anon.fit(G_anon)

# 获取嵌入
anon_nodes = sorted(G_anon.nodes())
orig_nodes = sorted(G.nodes())

anon_embeddings = model_anon.get_embeddings(anon_nodes)
orig_embeddings = model_orig.get_embeddings(orig_nodes)

print(f"嵌入维度: {anon_embeddings.shape}")

# ## 6. 去匿名化攻击

# ### 6.1 无种子节点攻击

matcher = EmbeddingMatcher()
similarity_matrix = matcher.compute_similarity_matrix(anon_embeddings, orig_embeddings)

predictions = matcher.match_greedy(similarity_matrix)

# 构建ground truth
gt_mapping = {}
for i, anon_node in enumerate(anon_nodes):
    orig_node = ground_truth['reverse_mapping'][anon_node]
    orig_idx = orig_nodes.index(orig_node)
    gt_mapping[i] = orig_idx

gt_list = [gt_mapping[i] for i in range(len(anon_nodes))]

# 评估
accuracy = calculate_accuracy(predictions, gt_mapping)
top_k = calculate_top_k_accuracy(similarity_matrix, gt_list, k_values=[1, 5, 10, 20])

print(f"\n无种子节点攻击:")
print(f"准确率: {accuracy:.4f}")
print(f"Top-5准确率: {top_k[5]:.4f}")
print(f"Top-10准确率: {top_k[10]:.4f}")

# ### 6.2 基于种子节点的攻击

# 随机选择5%的节点作为种子
n_seeds = int(len(anon_nodes) * 0.05)
seed_indices = np.random.choice(len(anon_nodes), n_seeds, replace=False)
seed_pairs = [(i, gt_mapping[i]) for i in seed_indices]

print(f"\n使用 {len(seed_pairs)} 个种子节点 ({n_seeds/len(anon_nodes)*100:.1f}%)")

# 图对齐
aligner = GraphAligner()
aligned_embeddings = aligner.align_procrustes(anon_embeddings, orig_embeddings, seed_pairs)

# 重新计算相似度
similarity_matrix_aligned = matcher.compute_similarity_matrix(aligned_embeddings, orig_embeddings)

# 匹配（带种子节点）
predictions_seed = matcher.match_with_seeds(similarity_matrix_aligned, seed_pairs)

# 评估
accuracy_seed = calculate_accuracy(predictions_seed, gt_mapping)
top_k_seed = calculate_top_k_accuracy(similarity_matrix_aligned, gt_list, k_values=[1, 5, 10, 20])

print(f"\n基于种子节点的攻击:")
print(f"准确率: {accuracy_seed:.4f}")
print(f"Top-5准确率: {top_k_seed[5]:.4f}")
print(f"Top-10准确率: {top_k_seed[10]:.4f}")

# ## 7. 结果可视化

# 对比不同方法
methods = ['无种子', '5%种子']
accuracies = [accuracy, accuracy_seed]

plt.figure(figsize=(10, 6))
plt.bar(methods, accuracies, alpha=0.7, color=['steelblue', 'coral'])
plt.ylabel('准确率 (Accuracy)', fontsize=12)
plt.title('去匿名化攻击效果对比', fontsize=14)
plt.ylim([0, 1])
plt.grid(alpha=0.3, axis='y')
plt.savefig('../results/attack_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Top-K对比
k_values = [1, 5, 10, 20]
plt.figure(figsize=(10, 6))
plt.plot(k_values, [top_k[k] for k in k_values], marker='o', label='无种子', linewidth=2)
plt.plot(k_values, [top_k_seed[k] for k in k_values], marker='s', label='5%种子', linewidth=2)
plt.xlabel('K', fontsize=12)
plt.ylabel('Top-K准确率', fontsize=12)
plt.title('Top-K准确率对比', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('../results/topk_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ## 8. 结论

print("\n" + "="*60)
print("实验结论:")
print("="*60)
print(f"1. 即使删除了{(1-ground_truth['edge_retention_ratio'])*100:.0f}%的边，")
print(f"   攻击者仍能以{accuracy*100:.1f}%的准确率识别匿名节点")
print(f"\n2. 使用{n_seeds}个种子节点({n_seeds/len(anon_nodes)*100:.1f}%)后，")
print(f"   攻击准确率提升至{accuracy_seed*100:.1f}%")
print(f"\n3. 这证明了'即便我不说话，我的朋友也会暴露我'")
print(f"   结构性隐私泄露是真实存在的风险！")
print("="*60)


