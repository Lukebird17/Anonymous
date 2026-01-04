"""
第一阶段完整原理演示 - 身份去匿名化
生成详细的演示数据，包括：
1. 贪心匹配：相似度矩阵、逐步选择过程
2. 匈牙利算法：成本矩阵变化、迭代过程
3. 图核方法：子图提取、WL核迭代
4. DeepWalk：随机游走路径、嵌入空间可视化
"""

import networkx as nx
import numpy as np
import json
from collections import defaultdict, Counter
from data.dataset_loader import DatasetLoader
from preprocessing.anonymizer import GraphAnonymizer
from models.feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import os


class Stage1PrincipleVisualizer:
    """第一阶段：身份去匿名化完整原理演示"""
    
    def __init__(self, ego_id='698'):
        self.ego_id = ego_id
        print(f"加载 Facebook Ego Network {ego_id}...")
        
        loader = DatasetLoader()
        self.G, self.attributes = loader.load_facebook(ego_network=ego_id)
        
        print(f"图规模: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 选择小规模子图以便清晰演示
        if self.G.number_of_nodes() > 30:
            print("选择核心子图用于演示...")
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:20]
            self.G = self.G.subgraph(top_nodes).copy()
            print(f"子图: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 生成匿名图
        print("生成匿名图...")
        anonymizer = GraphAnonymizer(self.G)
        self.G_anon, self.node_mapping = anonymizer.anonymize_with_perturbation(
            edge_retention_ratio=0.9,
            noise_edge_ratio=0.05
        )
        
        # 计算特征
        self.extractor = FeatureExtractor()
        self.nodes_orig = sorted(list(self.G.nodes()))
        self.nodes_anon = sorted(list(self.G_anon.nodes()))
        
        print("计算节点特征...")
        self.features_orig = self.extractor.extract_node_features(self.G, self.nodes_orig)
        self.features_anon = self.extractor.extract_node_features(self.G_anon, self.nodes_anon)
        
        self.scaler = StandardScaler()
        self.features_orig_scaled = self.scaler.fit_transform(self.features_orig)
        self.features_anon_scaled = self.scaler.transform(self.features_anon)
        
        self.similarity_matrix = cosine_similarity(self.features_orig_scaled, self.features_anon_scaled)
        
        print("准备ground truth...")
        self.ground_truth = {}
        for orig_node in self.nodes_orig:
            if orig_node in self.node_mapping:
                anon_node = self.node_mapping[orig_node]
                if anon_node in self.nodes_anon:
                    self.ground_truth[str(orig_node)] = str(anon_node)
    
    def method1_greedy_detailed(self):
        """方法1: 贪心匹配 - 详细步骤"""
        print("\n=== 方法1: 贪心特征匹配 ===")
        
        result = {
            'method': 'greedy',
            'nodes_orig': [str(n) for n in self.nodes_orig],
            'nodes_anon': [str(n) for n in self.nodes_anon],
            'features': {},
            'similarity_matrix': self.similarity_matrix.tolist(),
            'steps': [],
            'ground_truth': self.ground_truth
        }
        
        # 添加节点特征详情
        for i, node in enumerate(self.nodes_orig):
            result['features'][str(node)] = {
                'degree': int(self.G.degree(node)),
                'clustering': float(nx.clustering(self.G, node)),
                'triangles': int(sum(nx.triangles(self.G, [node]).values())),
                'neighbors': [str(n) for n in self.G.neighbors(node)]
            }
        
        for i, node in enumerate(self.nodes_anon):
            result['features'][f"anon_{node}"] = {
                'degree': int(self.G_anon.degree(node)),
                'clustering': float(nx.clustering(self.G_anon, node)),
                'triangles': int(sum(nx.triangles(self.G_anon, [node]).values())),
                'neighbors': [str(n) for n in self.G_anon.neighbors(node)]
            }
        
        # 贪心匹配过程
        sim_copy = self.similarity_matrix.copy()
        matched_orig = set()
        matched_anon = set()
        
        for step_idx in range(min(len(self.nodes_orig), 10)):  # 演示前10步
            # 找最大相似度
            max_sim = -1
            best_i, best_j = -1, -1
            
            for i in range(len(self.nodes_orig)):
                if i in matched_orig:
                    continue
                for j in range(len(self.nodes_anon)):
                    if j in matched_anon:
                        continue
                    if sim_copy[i][j] > max_sim:
                        max_sim = sim_copy[i][j]
                        best_i, best_j = i, j
            
            if best_i == -1:
                break
            
            orig_node = str(self.nodes_orig[best_i])
            anon_node = str(self.nodes_anon[best_j])
            
            # 找当前的top-5候选
            candidates = []
            for j in range(len(self.nodes_anon)):
                if j not in matched_anon:
                    candidates.append({
                        'node': str(self.nodes_anon[j]),
                        'similarity': float(self.similarity_matrix[best_i][j]),
                        'is_best': (j == best_j)
                    })
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            
            is_correct = (orig_node in self.ground_truth and 
                         self.ground_truth[orig_node] == anon_node)
            
            # 当前相似度矩阵状态（只显示未匹配的）
            current_matrix = []
            for i in range(len(self.nodes_orig)):
                if i not in matched_orig:
                    row = []
                    for j in range(len(self.nodes_anon)):
                        if j not in matched_anon:
                            row.append({
                                'value': float(sim_copy[i][j]),
                                'is_max': (i == best_i and j == best_j),
                                'orig_node': str(self.nodes_orig[i]),
                                'anon_node': str(self.nodes_anon[j])
                            })
                    current_matrix.append(row)
            
            step = {
                'step': step_idx + 1,
                'orig_node': orig_node,
                'anon_node': anon_node,
                'similarity': float(max_sim),
                'is_correct': is_correct,
                'candidates': candidates[:5],
                'matrix_state': current_matrix[:5],  # 只显示前5行
                'matched_count': len(matched_orig)
            }
            
            result['steps'].append(step)
            
            matched_orig.add(best_i)
            matched_anon.add(best_j)
            sim_copy[best_i, :] = -1
            sim_copy[:, best_j] = -1
        
        return result
    
    def method2_hungarian_detailed(self):
        """方法2: 匈牙利算法 - 详细步骤"""
        print("\n=== 方法2: 匈牙利算法 ===")
        
        result = {
            'method': 'hungarian',
            'nodes_orig': [str(n) for n in self.nodes_orig],
            'nodes_anon': [str(n) for n in self.nodes_anon],
            'cost_matrix_initial': (-self.similarity_matrix).tolist(),
            'steps': [],
            'final_matching': {},
            'ground_truth': self.ground_truth
        }
        
        # 求解匈牙利算法
        cost_matrix = -self.similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 记录最终匹配
        for i, j in zip(row_ind, col_ind):
            orig_node = str(self.nodes_orig[i])
            anon_node = str(self.nodes_anon[j])
            result['final_matching'][orig_node] = {
                'matched_node': anon_node,
                'similarity': float(self.similarity_matrix[i][j]),
                'is_correct': (orig_node in self.ground_truth and 
                              self.ground_truth[orig_node] == anon_node)
            }
        
        # 模拟匈牙利算法的关键步骤（简化版）
        # 步骤1: 行归约
        cost_reduced = cost_matrix.copy()
        for i in range(len(cost_reduced)):
            row_min = cost_reduced[i].min()
            cost_reduced[i] -= row_min
        
        result['steps'].append({
            'step': 1,
            'name': '行归约',
            'description': '每行减去该行最小值',
            'matrix': cost_reduced.tolist()
        })
        
        # 步骤2: 列归约
        for j in range(len(cost_reduced[0])):
            col_min = cost_reduced[:, j].min()
            cost_reduced[:, j] -= col_min
        
        result['steps'].append({
            'step': 2,
            'name': '列归约',
            'description': '每列减去该列最小值',
            'matrix': cost_reduced.tolist()
        })
        
        return result
    
    def method3_graph_kernel_detailed(self):
        """方法3: 图核方法 - 详细步骤"""
        print("\n=== 方法3: 图核方法 ===")
        
        result = {
            'method': 'graph_kernel',
            'nodes_orig': [str(n) for n in self.nodes_orig],
            'nodes_anon': [str(n) for n in self.nodes_anon],
            'subgraph_examples': [],
            'wl_iterations': [],
            'ground_truth': self.ground_truth
        }
        
        # 演示几个节点的子图提取
        demo_nodes = self.nodes_orig[:3]
        
        for node in demo_nodes:
            # 1-hop子图
            neighbors_1hop = list(self.G.neighbors(node))
            subgraph_1hop = self.G.subgraph([node] + neighbors_1hop)
            
            # 2-hop邻居
            neighbors_2hop = set()
            for n in neighbors_1hop:
                neighbors_2hop.update(self.G.neighbors(n))
            neighbors_2hop.discard(node)
            neighbors_2hop = list(neighbors_2hop)[:10]  # 限制数量
            
            subgraph_example = {
                'node': str(node),
                '1hop_neighbors': [str(n) for n in neighbors_1hop],
                '1hop_edges': [[str(u), str(v)] for u, v in subgraph_1hop.edges()],
                '2hop_neighbors': [str(n) for n in neighbors_2hop],
                'features': {
                    '1hop_nodes': len(neighbors_1hop) + 1,
                    '1hop_edges': subgraph_1hop.number_of_edges(),
                    '2hop_nodes': len(neighbors_2hop),
                    'avg_degree': sum([d for n, d in subgraph_1hop.degree()]) / (len(neighbors_1hop) + 1)
                }
            }
            
            result['subgraph_examples'].append(subgraph_example)
        
        # 模拟WL核的迭代过程
        # 初始标签：度数
        labels = {str(node): self.G.degree(node) for node in self.nodes_orig}
        
        result['wl_iterations'].append({
            'iteration': 0,
            'description': '初始化：使用节点度数作为标签',
            'labels': labels.copy()
        })
        
        # 迭代1-3次
        for iter_idx in range(3):
            new_labels = {}
            for node in self.nodes_orig:
                node_str = str(node)
                neighbors = list(self.G.neighbors(node))
                neighbor_labels = sorted([labels[str(n)] for n in neighbors])
                # 新标签 = hash(当前标签 + 排序后的邻居标签)
                new_label = hash((labels[node_str], tuple(neighbor_labels))) % 1000
                new_labels[node_str] = new_label
            
            labels = new_labels
            
            result['wl_iterations'].append({
                'iteration': iter_idx + 1,
                'description': f'迭代{iter_idx + 1}: 聚合邻居标签并更新',
                'labels': labels.copy()
            })
        
        return result
    
    def method4_deepwalk_detailed(self):
        """方法4: DeepWalk - 详细步骤"""
        print("\n=== 方法4: DeepWalk图嵌入 ===")
        
        result = {
            'method': 'deepwalk',
            'nodes_orig': [str(n) for n in self.nodes_orig],
            'nodes_anon': [str(n) for n in self.nodes_anon],
            'random_walks': [],
            'embedding_2d': {},
            'ground_truth': self.ground_truth
        }
        
        # 演示几个节点的随机游走
        demo_nodes = self.nodes_orig[:5]
        walk_length = 10
        num_walks = 3
        
        for node in demo_nodes:
            walks_from_node = []
            
            for walk_idx in range(num_walks):
                walk = [node]
                current = node
                
                for _ in range(walk_length - 1):
                    neighbors = list(self.G.neighbors(current))
                    if not neighbors:
                        break
                    current = np.random.choice(neighbors)
                    walk.append(current)
                
                walks_from_node.append({
                    'walk_id': walk_idx + 1,
                    'path': [str(n) for n in walk],
                    'length': len(walk)
                })
            
            result['random_walks'].append({
                'start_node': str(node),
                'walks': walks_from_node
            })
        
        # 模拟嵌入向量（用PCA降维到2D）
        from sklearn.decomposition import PCA
        
        # 使用实际的特征作为"嵌入"的近似
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.features_orig_scaled)
        
        for i, node in enumerate(self.nodes_orig):
            result['embedding_2d'][str(node)] = {
                'x': float(embeddings_2d[i][0]),
                'y': float(embeddings_2d[i][1]),
                'degree': int(self.G.degree(node))
            }
        
        # 匿名图的嵌入
        embeddings_anon_2d = pca.transform(self.features_anon_scaled)
        for i, node in enumerate(self.nodes_anon):
            result['embedding_2d'][f"anon_{node}"] = {
                'x': float(embeddings_anon_2d[i][0]),
                'y': float(embeddings_anon_2d[i][1]),
                'degree': int(self.G_anon.degree(node))
            }
        
        return result
    
    def graph_to_json(self, G):
        """转换图为JSON"""
        nodes = []
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            nodes.append({
                'id': str(node),
                'index': node_to_idx[node],
                'degree': G.degree(node)
            })
        
        edges = []
        for u, v in G.edges():
            edges.append({
                'source': node_to_idx[u],
                'target': node_to_idx[v]
            })
        
        return {'nodes': nodes, 'links': edges}
    
    def generate_all_data(self):
        """生成所有方法的数据"""
        print("\n" + "="*70)
        print("生成第一阶段所有方法的详细演示数据")
        print("="*70)
        
        data = {
            'graph_orig': self.graph_to_json(self.G),
            'graph_anon': self.graph_to_json(self.G_anon),
            'method1_greedy': self.method1_greedy_detailed(),
            'method2_hungarian': self.method2_hungarian_detailed(),
            'method3_graph_kernel': self.method3_graph_kernel_detailed(),
            'method4_deepwalk': self.method4_deepwalk_detailed()
        }
        
        return data
    
    def save_data(self, output_file='results/stage1_demo_data.json'):
        """保存数据到JSON文件"""
        data = self.generate_all_data()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 数据已保存到: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成第一阶段详细演示数据")
    parser.add_argument('--ego_id', type=str, default='698')
    parser.add_argument('--output', type=str, default='results/stage1_demo_data.json')
    
    args = parser.parse_args()
    
    print("="*70)
    print("第一阶段：身份去匿名化完整原理演示数据生成")
    print("="*70)
    
    visualizer = Stage1PrincipleVisualizer(ego_id=args.ego_id)
    output_file = visualizer.save_data(output_file=args.output)
    
    print("\n" + "="*70)
    print("✅ 数据生成完成！")
    print("\n包含内容:")
    print("  1. 贪心匹配：相似度矩阵、逐步选择过程、候选节点")
    print("  2. 匈牙利算法：成本矩阵、行列归约、最终匹配")
    print("  3. 图核方法：1-hop/2-hop子图、WL核迭代过程")
    print("  4. DeepWalk：随机游走路径、2D嵌入空间坐标")
    print("\n下一步：生成HTML可视化界面")
    print("="*70)


if __name__ == "__main__":
    main()








