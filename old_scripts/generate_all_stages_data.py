#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成所有三个阶段的完整演示数据
"""

import json
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

def create_demo_graph():
    """创建演示用的小图"""
    G = nx.karate_club_graph()
    # 只使用前20个节点
    nodes = list(G.nodes())[:20]
    G = G.subgraph(nodes).copy()
    
    # 重新编号
    mapping = {old: new for new, old in enumerate(nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    return G

def perturb_graph(G, retention_rate=0.8, noise_rate=0.1):
    """扰动图生成匿名图"""
    G_anon = G.copy()
    
    # 删除一些边
    edges = list(G_anon.edges())
    edges_to_remove = np.random.choice(len(edges), 
                                      size=int(len(edges) * (1-retention_rate)), 
                                      replace=False)
    for idx in edges_to_remove:
        G_anon.remove_edge(*edges[idx])
    
    # 添加噪声边
    nodes = list(G_anon.nodes())
    num_noise = int(len(edges) * noise_rate)
    for _ in range(num_noise):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G_anon.has_edge(u, v):
            G_anon.add_edge(u, v)
    
    return G_anon

def graph_to_json(G, name="orig"):
    """将图转换为JSON格式"""
    nodes = []
    for node in G.nodes():
        nodes.append({
            "id": str(node),
            "index": int(node),
            "degree": G.degree(node)
        })
    
    links = []
    for u, v in G.edges():
        links.append({
            "source": int(u),
            "target": int(v)
        })
    
    return {"nodes": nodes, "links": links}

# ==================== 第一阶段：身份去匿名化 ====================

def generate_stage1_greedy(G_orig, G_anon):
    """贪心匹配数据"""
    def node_features(G, node):
        """计算节点特征"""
        degree = G.degree(node)
        neighbors = list(G.neighbors(node))
        clustering = nx.clustering(G, node)
        triangles = sum(1 for n1 in neighbors for n2 in neighbors 
                       if n1 < n2 and G.has_edge(n1, n2)) // 3
        return np.array([degree, clustering, triangles])
    
    # 计算相似度矩阵
    nodes_orig = list(G_orig.nodes())[:10]
    nodes_anon = list(G_anon.nodes())[:10]
    
    similarity_matrix = []
    for i in nodes_orig:
        row = []
        for j in nodes_anon:
            f1 = node_features(G_orig, i)
            f2 = node_features(G_anon, j)
            # 余弦相似度
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-10)
            row.append(float(sim))
        similarity_matrix.append(row)
    
    # 贪心匹配步骤
    steps = []
    matrix_copy = [row[:] for row in similarity_matrix]
    matched_orig = set()
    matched_anon = set()
    
    for step in range(min(10, len(nodes_orig))):
        # 找最大值
        max_val = -1
        best_i, best_j = -1, -1
        for i in range(len(nodes_orig)):
            if i in matched_orig:
                continue
            for j in range(len(nodes_anon)):
                if j in matched_anon:
                    continue
                if matrix_copy[i][j] > max_val:
                    max_val = matrix_copy[i][j]
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        matched_orig.add(best_i)
        matched_anon.add(best_j)
        
        steps.append({
            "step": step + 1,
            "orig_node": int(nodes_orig[best_i]),
            "anon_node": int(nodes_anon[best_j]),
            "similarity": float(max_val),
            "is_correct": nodes_orig[best_i] == nodes_anon[best_j]
        })
    
    return {
        "similarity_matrix": similarity_matrix,
        "nodes_orig": [int(n) for n in nodes_orig],
        "nodes_anon": [int(n) for n in nodes_anon],
        "steps": steps
    }

def generate_stage1_hungarian(G_orig, G_anon):
    """匈牙利算法数据"""
    from scipy.optimize import linear_sum_assignment
    
    def node_features(G, node):
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        return np.array([degree, clustering])
    
    nodes_orig = list(G_orig.nodes())[:10]
    nodes_anon = list(G_anon.nodes())[:10]
    
    # 相似度矩阵
    sim_matrix = []
    for i in nodes_orig:
        row = []
        for j in nodes_anon:
            f1 = node_features(G_orig, i)
            f2 = node_features(G_anon, j)
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-10)
            row.append(float(sim))
        sim_matrix.append(row)
    
    # 转换为成本矩阵
    cost_matrix_initial = [[-s for s in row] for row in sim_matrix]
    
    # 行归约
    cost_after_row = []
    for row in cost_matrix_initial:
        min_val = min(row)
        cost_after_row.append([c - min_val for c in row])
    
    # 列归约
    cost_after_col = [row[:] for row in cost_after_row]
    for j in range(len(cost_after_col[0])):
        min_val = min(row[j] for row in cost_after_col)
        for i in range(len(cost_after_col)):
            cost_after_col[i][j] -= min_val
    
    # 求解
    row_ind, col_ind = linear_sum_assignment(cost_after_col)
    
    final_matching = {}
    for i, j in zip(row_ind, col_ind):
        final_matching[str(nodes_orig[i])] = {
            "matched_node": int(nodes_anon[j]),
            "similarity": float(sim_matrix[i][j]),
            "is_correct": nodes_orig[i] == nodes_anon[j]
        }
    
    return {
        "cost_matrix_initial": cost_matrix_initial,
        "steps": [
            {"step": 1, "name": "初始成本矩阵", "description": "将相似度转换为成本（取负）", 
             "matrix": cost_matrix_initial},
            {"step": 2, "name": "行归约", "description": "每行减去该行最小值", 
             "matrix": cost_after_row},
            {"step": 3, "name": "列归约", "description": "每列减去该列最小值", 
             "matrix": cost_after_col}
        ],
        "final_matching": final_matching
    }

def generate_stage1_graph_kernel(G_orig, G_anon):
    """图核方法数据"""
    nodes = list(G_orig.nodes())[:5]
    
    subgraph_examples = []
    for node in nodes[:3]:
        neighbors_1hop = list(G_orig.neighbors(node))
        edges_1hop = [(int(u), int(v)) for u, v in G_orig.edges(node)]
        
        subgraph_examples.append({
            "node": int(node),
            "1hop_neighbors": [int(n) for n in neighbors_1hop[:5]],
            "1hop_edges": edges_1hop[:10]
        })
    
    # WL核迭代
    wl_iterations = []
    labels = {n: G_orig.degree(n) for n in G_orig.nodes()}
    
    wl_iterations.append({
        "iteration": 0,
        "description": "初始化：标签 = 节点度数",
        "labels": {str(k): int(v) for k, v in list(labels.items())[:10]}
    })
    
    for iter_num in range(1, 4):
        new_labels = {}
        for node in G_orig.nodes():
            neighbor_labels = sorted([labels[n] for n in G_orig.neighbors(node)])
            new_label = hash((labels[node], tuple(neighbor_labels))) % 10000
            new_labels[node] = new_label
        
        labels = new_labels
        wl_iterations.append({
            "iteration": iter_num,
            "description": f"迭代{iter_num}：聚合邻居标签",
            "labels": {str(k): int(v) for k, v in list(labels.items())[:10]}
        })
    
    return {
        "subgraph_examples": subgraph_examples,
        "wl_iterations": wl_iterations
    }

def generate_stage1_deepwalk(G_orig, G_anon):
    """DeepWalk数据"""
    nodes = list(G_orig.nodes())[:10]
    
    # 生成随机游走
    random_walks = []
    for node in nodes[:5]:
        walks = []
        for _ in range(3):
            walk = [node]
            current = node
            for _ in range(10):
                neighbors = list(G_orig.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                walk.append(current)
            walks.append({"path": [int(n) for n in walk]})
        
        random_walks.append({
            "start_node": int(node),
            "walks": walks
        })
    
    # 模拟嵌入（实际应该用Skip-gram训练）
    embedding_2d = {}
    for node in G_orig.nodes():
        # 使用节点度数和聚类系数作为简单特征
        x = G_orig.degree(node) + np.random.randn() * 0.5
        y = nx.clustering(G_orig, node) * 10 + np.random.randn() * 0.5
        embedding_2d[str(node)] = {"x": float(x), "y": float(y)}
    
    for node in G_anon.nodes():
        x = G_anon.degree(node) + np.random.randn() * 0.5
        y = nx.clustering(G_anon, node) * 10 + np.random.randn() * 0.5
        embedding_2d[f"anon_{node}"] = {"x": float(x), "y": float(y)}
    
    return {
        "random_walks": random_walks,
        "embedding_2d": embedding_2d
    }

# ==================== 第二阶段：属性推断 ====================

def generate_stage2_neighbor_voting(G_orig):
    """邻居投票数据"""
    # 模拟一些节点有标签
    nodes = list(G_orig.nodes())
    num_labeled = len(nodes) // 2
    
    # 随机分配标签（0或1）
    true_labels = {n: np.random.randint(0, 2) for n in nodes}
    known_labels = {n: true_labels[n] for n in nodes[:num_labeled]}
    
    # 对未知节点进行投票
    steps = []
    for target_node in nodes[num_labeled:num_labeled+5]:
        neighbors = list(G_orig.neighbors(target_node))
        known_neighbors = [n for n in neighbors if n in known_labels]
        
        if not known_neighbors:
            continue
        
        votes = [known_labels[n] for n in known_neighbors]
        vote_0 = votes.count(0)
        vote_1 = votes.count(1)
        predicted = 0 if vote_0 > vote_1 else 1
        
        steps.append({
            "target_node": int(target_node),
            "neighbors": [int(n) for n in known_neighbors],
            "votes": {"label_0": vote_0, "label_1": vote_1},
            "predicted": predicted,
            "actual": int(true_labels[target_node]),
            "is_correct": predicted == true_labels[target_node]
        })
    
    return {
        "known_labels": {str(k): int(v) for k, v in known_labels.items()},
        "steps": steps
    }

def generate_stage2_label_propagation(G_orig):
    """标签传播数据"""
    nodes = list(G_orig.nodes())
    num_labeled = len(nodes) // 3
    
    # 初始标签
    true_labels = {n: np.random.randint(0, 2) for n in nodes}
    labels = {n: true_labels[n] for n in nodes[:num_labeled]}
    
    iterations = []
    iterations.append({
        "iteration": 0,
        "description": "初始状态：部分节点有标签",
        "labels": {str(k): int(v) for k, v in list(labels.items())[:10]},
        "num_labeled": len(labels)
    })
    
    # 传播3轮
    for iter_num in range(1, 4):
        new_labels = labels.copy()
        for node in nodes:
            if node in labels:
                continue
            
            neighbors = list(G_orig.neighbors(node))
            neighbor_labels = [labels[n] for n in neighbors if n in labels]
            
            if neighbor_labels:
                # 多数投票
                predicted = Counter(neighbor_labels).most_common(1)[0][0]
                new_labels[node] = predicted
        
        labels = new_labels
        iterations.append({
            "iteration": iter_num,
            "description": f"迭代{iter_num}：标签传播到邻居",
            "labels": {str(k): int(v) for k, v in list(labels.items())[:10]},
            "num_labeled": len(labels)
        })
    
    return {"iterations": iterations}

def generate_stage2_graphsage(G_orig):
    """GraphSAGE数据（简化版）"""
    nodes = list(G_orig.nodes())[:10]
    
    # 模拟聚合过程
    aggregation_steps = []
    for node in nodes[:3]:
        neighbors = list(G_orig.neighbors(node))[:5]
        
        # 特征聚合
        aggregation_steps.append({
            "target_node": int(node),
            "neighbors": [int(n) for n in neighbors],
            "description": f"聚合节点{node}的邻居特征",
            "aggregation_type": "mean"
        })
    
    return {
        "aggregation_steps": aggregation_steps,
        "description": "GraphSAGE通过采样和聚合邻居特征来学习节点表示"
    }

# ==================== 第三阶段：鲁棒性与防御 ====================

def generate_stage3_robustness(G_orig):
    """鲁棒性测试数据"""
    # 测试不同的边保留率
    retention_rates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    accuracy_results = []
    
    for rate in retention_rates:
        # 模拟准确率（实际应该运行攻击）
        # 准确率随着边减少而下降
        accuracy = 0.9 * rate + np.random.randn() * 0.05
        accuracy = max(0.1, min(1.0, accuracy))
        
        accuracy_results.append({
            "retention_rate": rate,
            "accuracy": float(accuracy),
            "edges_kept": int(G_orig.number_of_edges() * rate)
        })
    
    return {
        "test_name": "边保留率测试",
        "description": "测试在边被移除的情况下，攻击的准确率",
        "results": accuracy_results
    }

def generate_stage3_defense(G_orig):
    """差分隐私防御数据"""
    # 测试不同的epsilon值
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    defense_results = []
    
    for epsilon in epsilon_values:
        # 模拟防御效果
        # epsilon越小，隐私保护越强，但准确率越低
        attack_accuracy = 1.0 / (1 + np.exp(-epsilon + 2)) + np.random.randn() * 0.05
        attack_accuracy = max(0.1, min(1.0, attack_accuracy))
        
        defense_results.append({
            "epsilon": epsilon,
            "attack_accuracy": float(attack_accuracy),
            "privacy_level": "高" if epsilon < 1.0 else ("中" if epsilon < 5.0 else "低")
        })
    
    return {
        "defense_name": "差分隐私边扰动",
        "description": "通过添加满足差分隐私的噪声边来保护隐私",
        "results": defense_results
    }

def main():
    print("生成所有阶段的演示数据...")
    
    np.random.seed(42)
    
    # 创建图
    G_orig = create_demo_graph()
    G_anon = perturb_graph(G_orig)
    
    print(f"原始图: {G_orig.number_of_nodes()} 节点, {G_orig.number_of_edges()} 边")
    print(f"匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
    
    data = {}
    
    # 图数据
    data["graph_orig"] = graph_to_json(G_orig)
    data["graph_anon"] = graph_to_json(G_anon)
    
    # 第一阶段
    print("\n生成第一阶段数据...")
    data["stage1"] = {
        "greedy": generate_stage1_greedy(G_orig, G_anon),
        "hungarian": generate_stage1_hungarian(G_orig, G_anon),
        "graph_kernel": generate_stage1_graph_kernel(G_orig, G_anon),
        "deepwalk": generate_stage1_deepwalk(G_orig, G_anon)
    }
    
    # 第二阶段
    print("生成第二阶段数据...")
    data["stage2"] = {
        "neighbor_voting": generate_stage2_neighbor_voting(G_orig),
        "label_propagation": generate_stage2_label_propagation(G_orig),
        "graphsage": generate_stage2_graphsage(G_orig)
    }
    
    # 第三阶段
    print("生成第三阶段数据...")
    data["stage3"] = {
        "robustness": generate_stage3_robustness(G_orig),
        "defense": generate_stage3_defense(G_orig)
    }
    
    # 保存
    output_path = "results/all_stages_demo_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"文件大小: {len(json.dumps(data)) / 1024:.1f} KB")
    
    print("\n包含的数据:")
    print("  第一阶段 (4个方法):")
    print("    - 贪心匹配")
    print("    - 匈牙利算法")
    print("    - 图核方法")
    print("    - DeepWalk")
    print("  第二阶段 (3个方法):")
    print("    - 邻居投票")
    print("    - 标签传播")
    print("    - GraphSAGE")
    print("  第三阶段 (2个测试):")
    print("    - 鲁棒性测试")
    print("    - 差分隐私防御")

if __name__ == "__main__":
    main()








