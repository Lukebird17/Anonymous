"""
完整原理演示版本 - 展示算法执行过程
不只是显示结果，而是演示每个方法的工作原理
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
import os


class PrincipleVisualizer:
    """原理演示可视化器"""
    
    def __init__(self, ego_id='698'):
        self.ego_id = ego_id
        print(f"加载 Facebook Ego Network {ego_id}...")
        
        loader = DatasetLoader()
        self.G, self.attributes = loader.load_facebook(ego_network=ego_id)
        
        print(f"图规模: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 如果网络太大，选择子图
        if self.G.number_of_nodes() > 80:
            print("网络较大，选择核心子图...")
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:50]
            self.G = self.G.subgraph(top_nodes).copy()
            print(f"子图规模: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 生成匿名图
        print("生成匿名图...")
        anonymizer = GraphAnonymizer(self.G)
        self.G_anon, self.node_mapping = anonymizer.anonymize_with_perturbation(
            edge_retention_ratio=0.9,
            noise_edge_ratio=0.05
        )
        
        # 准备标签
        self.prepare_labels()
        
    def prepare_labels(self):
        """准备标签数据"""
        self.node_labels = {}
        circle_to_int = {}
        next_label = 0
        
        for node in self.G.nodes():
            if node in self.attributes and 'circles' in self.attributes[node]:
                circles = self.attributes[node]['circles']
                if circles:
                    circle = circles[0]
                    if circle not in circle_to_int:
                        circle_to_int[circle] = next_label
                        next_label += 1
                    self.node_labels[node] = circle_to_int[circle]
        
        if not self.node_labels:
            degrees = dict(self.G.degree())
            for node in self.G.nodes():
                deg = degrees[node]
                if deg < 5:
                    self.node_labels[node] = 0
                elif deg < 15:
                    self.node_labels[node] = 1
                else:
                    self.node_labels[node] = 2
    
    def graph_to_json(self, G, include_labels=False):
        """转换图为JSON"""
        nodes = []
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            node_data = {
                'id': str(node),
                'index': node_to_idx[node],
                'degree': G.degree(node),
                'clustering': float(nx.clustering(G, node))
            }
            
            if include_labels and node in self.node_labels:
                node_data['label'] = int(self.node_labels[node])
            
            nodes.append(node_data)
        
        edges = []
        for u, v in G.edges():
            edges.append({
                'source': node_to_idx[u],
                'target': node_to_idx[v]
            })
        
        return {'nodes': nodes, 'links': edges}
    
    def prepare_greedy_matching_steps(self):
        """准备贪心匹配的详细步骤"""
        print("\n准备贪心匹配详细步骤...")
        
        extractor = FeatureExtractor()
        nodes_orig = sorted(list(self.G.nodes()))[:10]  # 只演示前10个节点
        nodes_anon = sorted(list(self.G_anon.nodes()))[:10]
        
        features_orig = extractor.extract_node_features(self.G, nodes_orig)
        features_anon = extractor.extract_node_features(self.G_anon, nodes_anon)
        
        scaler = StandardScaler()
        features_orig = scaler.fit_transform(features_orig)
        features_anon = scaler.transform(features_anon)
        
        similarity = cosine_similarity(features_orig, features_anon)
        
        # 准备ground truth
        ground_truth = {}
        for orig_node in nodes_orig:
            if orig_node in self.node_mapping:
                anon_node = self.node_mapping[orig_node]
                if anon_node in nodes_anon:
                    ground_truth[str(orig_node)] = str(anon_node)
        
        # 模拟贪心匹配过程
        steps = []
        matched_orig = set()
        matched_anon = set()
        
        similarity_copy = similarity.copy()
        
        for step_idx in range(min(5, len(nodes_orig))):  # 演示5步
            # 找最大相似度
            max_val = -1
            best_i, best_j = -1, -1
            
            for i in range(len(nodes_orig)):
                if i in matched_orig:
                    continue
                for j in range(len(nodes_anon)):
                    if j in matched_anon:
                        continue
                    if similarity_copy[i][j] > max_val:
                        max_val = similarity_copy[i][j]
                        best_i, best_j = i, j
            
            if best_i == -1:
                break
            
            orig_node = str(nodes_orig[best_i])
            anon_node = str(nodes_anon[best_j])
            
            # 获取该节点的top-5候选
            candidates = []
            for j in range(len(nodes_anon)):
                if j not in matched_anon:
                    candidates.append({
                        'node': str(nodes_anon[j]),
                        'similarity': float(similarity[best_i][j])
                    })
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            candidates = candidates[:5]
            
            is_correct = (orig_node in ground_truth and 
                         ground_truth[orig_node] == anon_node)
            
            step = {
                'step': step_idx + 1,
                'orig_node': orig_node,
                'orig_features': {
                    'degree': int(self.G.degree(int(orig_node))),
                    'clustering': float(nx.clustering(self.G, int(orig_node)))
                },
                'matched_node': anon_node,
                'similarity': float(max_val),
                'candidates': candidates,
                'is_correct': is_correct,
                'similarity_matrix': similarity_copy.tolist()
            }
            
            steps.append(step)
            
            matched_orig.add(best_i)
            matched_anon.add(best_j)
            
            # 将已匹配的设为-1
            similarity_copy[best_i, :] = -1
            similarity_copy[:, best_j] = -1
        
        return {
            'steps': steps,
            'nodes_orig': [str(n) for n in nodes_orig],
            'nodes_anon': [str(n) for n in nodes_anon],
            'ground_truth': ground_truth
        }
    
    def prepare_neighbor_voting_steps(self):
        """准备邻居投票的详细步骤"""
        print("\n准备邻居投票详细步骤...")
        
        if not self.node_labels:
            return None
        
        nodes_list = list(self.node_labels.keys())
        np.random.seed(42)
        nodes_to_hide = set(np.random.choice(nodes_list, len(nodes_list) // 2, replace=False))
        
        known_labels = {str(n): int(self.node_labels[n]) 
                       for n in nodes_list if n not in nodes_to_hide}
        hidden_labels = {str(n): int(self.node_labels[n]) 
                        for n in nodes_to_hide}
        
        # 演示5个节点的投票过程
        demo_nodes = list(nodes_to_hide)[:5]
        steps = []
        
        for idx, test_node in enumerate(demo_nodes):
            neighbors = list(self.G.neighbors(test_node))
            neighbor_labels = []
            neighbor_details = []
            
            for n in neighbors:
                if n in self.node_labels and n not in nodes_to_hide:
                    label = self.node_labels[n]
                    neighbor_labels.append(label)
                    neighbor_details.append({
                        'node': str(n),
                        'label': int(label)
                    })
            
            if neighbor_labels:
                label_counts = Counter(neighbor_labels)
                prediction = int(label_counts.most_common(1)[0][0])
                votes = {int(k): int(v) for k, v in label_counts.items()}
            else:
                prediction = 0
                votes = {}
            
            actual = int(self.node_labels[test_node])
            is_correct = (prediction == actual)
            
            step = {
                'step': idx + 1,
                'target_node': str(test_node),
                'neighbors': neighbor_details,
                'votes': votes,
                'prediction': prediction,
                'actual': actual,
                'is_correct': is_correct
            }
            
            steps.append(step)
        
        return {
            'steps': steps,
            'known_labels': known_labels,
            'hidden_labels': hidden_labels
        }
    
    def prepare_label_propagation_steps(self):
        """准备标签传播的详细步骤"""
        print("\n准备标签传播详细步骤...")
        
        if not self.node_labels:
            return None
        
        nodes_list = list(self.node_labels.keys())
        np.random.seed(42)
        nodes_to_hide = set(np.random.choice(nodes_list, len(nodes_list) // 2, replace=False))
        
        G_copy = self.G.copy()
        
        # 初始化标签
        for node in G_copy.nodes():
            if node not in nodes_to_hide:
                G_copy.nodes[node]['label'] = self.node_labels.get(node)
            else:
                G_copy.nodes[node]['label'] = None
        
        iterations = []
        
        for iteration in range(5):  # 最多5次迭代
            updated_nodes = []
            
            for test_node in nodes_to_hide:
                if G_copy.nodes[test_node]['label'] is None:
                    neighbors = list(G_copy.neighbors(test_node))
                    neighbor_labels = [G_copy.nodes[n]['label'] for n in neighbors 
                                     if G_copy.nodes[n]['label'] is not None]
                    
                    if neighbor_labels:
                        label_counts = Counter(neighbor_labels)
                        most_common = label_counts.most_common(1)[0][0]
                        G_copy.nodes[test_node]['label'] = most_common
                        
                        updated_nodes.append({
                            'node': str(test_node),
                            'new_label': int(most_common),
                            'votes': {int(k): int(v) for k, v in label_counts.items()},
                            'neighbors': [str(n) for n in neighbors if G_copy.nodes[n]['label'] is not None]
                        })
            
            if not updated_nodes:
                break
            
            iterations.append({
                'iteration': iteration + 1,
                'updated_nodes': updated_nodes
            })
        
        return {
            'iterations': iterations,
            'initial_known': {str(n): int(self.node_labels[n]) for n in nodes_list if n not in nodes_to_hide},
            'initial_unknown': [str(n) for n in nodes_to_hide]
        }
    
    def prepare_robustness_incremental(self):
        """准备鲁棒性测试的增量数据"""
        print("\n准备鲁棒性测试增量数据...")
        
        all_edges = list(self.G.edges())
        np.random.seed(42)
        
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        incremental_data = []
        
        cumulative_removed = []
        
        for ratio in missing_ratios:
            num_to_remove = int(len(all_edges) * ratio)
            edges_to_remove_indices = np.random.choice(
                len(all_edges), num_to_remove, replace=False
            )
            
            removed_edges = []
            for idx in edges_to_remove_indices:
                u, v = all_edges[idx]
                edge_dict = {'source': str(u), 'target': str(v)}
                removed_edges.append(edge_dict)
            
            # 找出本次新增的移除边
            new_removed = [e for e in removed_edges if e not in cumulative_removed]
            
            incremental_data.append({
                'ratio': ratio,
                'new_removed': new_removed,
                'total_removed': len(removed_edges),
                'remaining': len(all_edges) - len(removed_edges)
            })
            
            cumulative_removed = removed_edges.copy()
        
        return incremental_data
    
    def prepare_defense_incremental(self):
        """准备防御机制的增量数据"""
        print("\n准备防御机制增量数据...")
        
        nodes = list(self.G.nodes())
        np.random.seed(42)
        
        epsilon_values = [0.5, 1.0, 2.0, 5.0]
        incremental_data = []
        
        cumulative_added = []
        
        for epsilon in epsilon_values:
            num_noise_edges = int(self.G.number_of_edges() * 0.1 / epsilon)
            
            added_edges = []
            attempts = 0
            while len(added_edges) < num_noise_edges and attempts < num_noise_edges * 10:
                u = np.random.choice(nodes)
                v = np.random.choice(nodes)
                if u != v and not self.G.has_edge(u, v):
                    edge_dict = {'source': str(u), 'target': str(v)}
                    if edge_dict not in added_edges and edge_dict not in cumulative_added:
                        added_edges.append(edge_dict)
                attempts += 1
            
            # 本次新增的边
            new_added = [e for e in added_edges if e not in cumulative_added]
            
            privacy_level = '强' if epsilon < 1 else '中' if epsilon < 2 else '弱'
            
            incremental_data.append({
                'epsilon': epsilon,
                'new_added': new_added,
                'total_added': len(cumulative_added) + len(new_added),
                'privacy_level': privacy_level
            })
            
            cumulative_added.extend(new_added)
        
        return incremental_data
    
    def generate_html(self, output_file="results/attack_principles_complete.html"):
        """生成完整的原理演示HTML"""
        print("\n生成完整原理演示HTML...")
        
        # 准备所有数据
        graph_orig = self.graph_to_json(self.G, include_labels=True)
        graph_anon = self.graph_to_json(self.G_anon)
        
        greedy_data = self.prepare_greedy_matching_steps()
        voting_data = self.prepare_neighbor_voting_steps()
        propagation_data = self.prepare_label_propagation_steps()
        robustness_data = self.prepare_robustness_incremental()
        defense_data = self.prepare_defense_incremental()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 生成HTML（这里只是框架，完整的HTML会很长）
        html_content = self._generate_complete_html(
            graph_orig, graph_anon, greedy_data, voting_data,
            propagation_data, robustness_data, defense_data
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML已生成: {output_file}")
        return output_file
    
    def _generate_complete_html(self, graph_orig, graph_anon, greedy_data, 
                                voting_data, propagation_data, robustness_data, defense_data):
        """生成完整HTML - 包含所有原理演示"""
        
        # 将Python数据转为JSON字符串
        graph_orig_json = json.dumps(graph_orig)
        graph_anon_json = json.dumps(graph_anon)
        greedy_data_json = json.dumps(greedy_data)
        voting_data_json = json.dumps(voting_data) if voting_data else 'null'
        propagation_data_json = json.dumps(propagation_data) if propagation_data else 'null'
        robustness_data_json = json.dumps(robustness_data)
        defense_data_json = json.dumps(defense_data)
        
        # 导入HTML模板
        with open('visualize_complete_template.py', 'r') as f:
            template_content = f.read()
            # 提取HTML字符串部分
            start = template_content.find('html = f"""')
            end = template_content.rfind('"""', start) + 3
            html_template = template_content[start:end]
            
            # 执行f-string替换
            html = eval(html_template[10:])  # 去掉 'html = f'
        
        return html


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成完整原理演示")
    parser.add_argument('--ego_id', type=str, default='698')
    parser.add_argument('--output', type=str, default='results/attack_principles_complete.html')
    
    args = parser.parse_args()
    
    print("="*70)
    print("完整原理演示系统生成器")
    print("="*70)
    
    visualizer = PrincipleVisualizer(ego_id=args.ego_id)
    output_file = visualizer.generate_html(output_file=args.output)
    
    print("\n" + "="*70)
    print("✅ 生成完成！")
    print(f"📂 文件: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()

        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图去匿名化攻击原理完整演示系统</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            height: 100vh;
            overflow: hidden;
        }}
        
        .main-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 1.6em;
            margin-bottom: 5px;
        }}
        
        header p {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .content-wrapper {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        
        .graphs-panel {{
            flex: 0 0 60%;
            display: flex;
            flex-direction: column;
            padding: 15px;
            gap: 15px;
            overflow-y: auto;
        }}
        
        .graph-container {{
            flex: 1;
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            min-height: 350px;
        }}
        
        .graph-container h3 {{
            margin-bottom: 10px;
            color: #495057;
            font-size: 1.05em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        
        .graph-svg {{
            width: 100%;
            height: calc(100% - 45px);
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #f8f9fa;
        }}
        
        .control-panel {{
            flex: 0 0 40%;
            background: white;
            border-left: 3px solid #e9ecef;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }}
        
        .phase-selector {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .phase-selector h2 {{
            font-size: 1.1em;
            margin-bottom: 12px;
            color: #495057;
        }}
        
        .phase-buttons {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .phase-btn {{
            padding: 10px 15px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            background: white;
            color: #495057;
            font-size: 0.95em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: left;
        }}
        
        .phase-btn:hover {{
            background: #f8f9fa;
            border-color: #667eea;
        }}
        
        .phase-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .method-selector {{
            padding: 12px 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .method-selector h3 {{
            font-size: 0.95em;
            margin-bottom: 8px;
            color: #6c757d;
        }}
        
        .method-buttons {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        
        .method-btn {{
            padding: 8px 12px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: white;
            color: #495057;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }}
        
        .method-btn:hover {{
            background: #f8f9fa;
        }}
        
        .method-btn.active {{
            background: #e7f3ff;
            border-color: #667eea;
            color: #667eea;
            font-weight: 600;
        }}
        
        .demo-content {{
            flex: 1;
            padding: 15px 20px;
            overflow-y: auto;
        }}
        
        .explanation {{
            background: #fff3cd;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
            border-left: 4px solid #ffc107;
        }}
        
        .explanation h4 {{
            margin-bottom: 6px;
            color: #856404;
            font-size: 1em;
        }}
        
        .explanation p {{
            color: #856404;
            line-height: 1.5;
            font-size: 0.9em;
        }}
        
        .principle-box {{
            background: #e7f3ff;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
            border-left: 4px solid #007bff;
        }}
        
        .principle-box h4 {{
            margin-bottom: 8px;
            color: #004085;
            font-size: 0.95em;
        }}
        
        .principle-box .formula {{
            background: white;
            padding: 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin: 6px 0;
            overflow-x: auto;
        }}
        
        .principle-box ul {{
            margin-left: 20px;
            font-size: 0.9em;
            color: #004085;
        }}
        
        .steps-container {{
            margin-top: 12px;
        }}
        
        .steps-container h3 {{
            font-size: 1em;
            margin-bottom: 10px;
            color: #495057;
        }}
        
        .step {{
            background: white;
            padding: 10px;
            border-radius: 6px;
            margin: 8px 0;
            border-left: 3px solid #28a745;
            font-size: 0.85em;
            transition: all 0.3s ease;
        }}
        
        .step.current {{
            background: #e7f3ff;
            border-left-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
            transform: translateX(5px);
        }}
        
        .step strong {{
            color: #495057;
        }}
        
        .step .detail {{
            margin-top: 6px;
            padding: 6px;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .step .votes {{
            display: flex;
            gap: 8px;
            margin-top: 6px;
            flex-wrap: wrap;
        }}
        
        .vote-badge {{
            background: #667eea;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.85em;
        }}
        
        .controls {{
            padding: 12px 20px;
            background: #f8f9fa;
            border-top: 2px solid #e9ecef;
            display: flex;
            gap: 8px;
        }}
        
        .control-btn {{
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .control-btn.play {{
            background: #28a745;
            color: white;
        }}
        
        .control-btn.play:hover {{
            background: #218838;
        }}
        
        .control-btn.next {{
            background: #007bff;
            color: white;
        }}
        
        .control-btn.next:hover {{
            background: #0056b3;
        }}
        
        .control-btn.reset {{
            background: #6c757d;
            color: white;
        }}
        
        .control-btn.reset:hover {{
            background: #5a6268;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 12px;
        }}
        
        .stat-card {{
            background: white;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.8em;
            margin-top: 4px;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8em;
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid #333;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .node:hover {{
            stroke-width: 3px;
        }}
        
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5;
        }}
        
        .link.removed {{
            stroke: #ff6b6b;
            stroke-opacity: 0.4;
            stroke-dasharray: 5,5;
            stroke-width: 2;
        }}
        
        .link.added {{
            stroke: #51cf66;
            stroke-opacity: 0.8;
            stroke-width: 2.5;
        }}
        
        .node.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 4px;
            r: 8;
        }}
        
        .node.target {{
            stroke: #ffd43b;
            stroke-width: 4px;
            r: 8;
        }}
        
        .node.neighbor {{
            stroke: #51cf66;
            stroke-width: 3px;
        }}
        
        .node.matched {{
            stroke: #51cf66;
            stroke-width: 3px;
        }}
        
        .node.candidate {{
            stroke: #ffd43b;
            stroke-width: 3px;
        }}
        
        .tooltip {{
            position: absolute;
            padding: 8px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            font-size: 0.85em;
            z-index: 1000;
            display: none;
        }}
        
        .similarity-matrix {{
            width: 100%;
            margin: 10px 0;
            font-size: 0.75em;
        }}
        
        .similarity-matrix td {{
            padding: 3px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        
        .similarity-matrix .high {{
            background: #51cf66;
            color: white;
        }}
        
        .similarity-matrix .medium {{
            background: #ffd43b;
        }}
        
        .similarity-matrix .low {{
            background: #f8f9fa;
        }}
        
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 3px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        
        .live-server-notice {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: #ffc107;
            color: #000;
            padding: 10px 15px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 10000;
            font-size: 0.85em;
            max-width: 280px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <header>
            <h1>🔍 图去匿名化攻击原理完整演示系统</h1>
            <p>深入理解算法执行过程 | 数据集: Facebook Ego {self.ego_id} ({self.G.number_of_nodes()}节点, {self.G.number_of_edges()}边)</p>
        </header>
        
        <div class="content-wrapper">
            <!-- 左侧：图可视化 -->
            <div class="graphs-panel">
                <div class="graph-container">
                    <h3 id="graph-top-title">原始图</h3>
                    <svg id="graph-top" class="graph-svg"></svg>
                </div>
                
                <div class="graph-container">
                    <h3 id="graph-bottom-title">匿名图/处理后的图</h3>
                    <svg id="graph-bottom" class="graph-svg"></svg>
                </div>
            </div>
            
            <!-- 右侧：控制和原理展示 -->
            <div class="control-panel">
                <div class="phase-selector">
                    <h2>选择攻击阶段</h2>
                    <div class="phase-buttons">
                        <button class="phase-btn active" data-phase="deanonymization">
                            🎯 阶段1: 身份去匿名化
                        </button>
                        <button class="phase-btn" data-phase="attribute">
                            🏷️ 阶段2: 属性推断
                        </button>
                        <button class="phase-btn" data-phase="robustness">
                            🛡️ 阶段3: 鲁棒性测试
                        </button>
                        <button class="phase-btn" data-phase="defense">
                            🔒 阶段4: 防御机制
                        </button>
                    </div>
                </div>
                
                <div class="method-selector">
                    <h3>选择演示方法</h3>
                    <div id="method-buttons" class="method-buttons"></div>
                </div>
                
                <div class="demo-content">
                    <div id="principle" class="principle-box" style="display:none;"></div>
                    <div id="explanation" class="explanation"></div>
                    <div id="steps-container" class="steps-container"></div>
                    
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #4dabf7;"></div>
                            <span>普通节点</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff6b6b;"></div>
                            <span>当前处理</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffd43b;"></div>
                            <span>候选/邻居</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #51cf66;"></div>
                            <span>成功/正确</span>
                        </div>
                    </div>
                    
                    <div id="stats" class="stats"></div>
                </div>
                
                <div class="controls">
                    <button class="control-btn play" id="play-btn">▶️ 开始</button>
                    <button class="control-btn next" id="next-btn">⏭️ 下一步</button>
                    <button class="control-btn reset" id="reset-btn">🔄 重置</button>
                </div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // 嵌入数据
        const DATA = {{
            graphOrig: {graph_orig_json},
            graphAnon: {graph_anon_json},
            greedyMatching: {greedy_data_json},
            neighborVoting: {voting_data_json},
            labelPropagation: {propagation_data_json},
            robustness: {robustness_data_json},
            defense: {defense_data_json}
        }};
        
        // 全局状态
        let currentPhase = 'deanonymization';
        let currentMethod = null;
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        
        // 图表实例
        let topChart = null;
        let bottomChart = null;
        
        // Live Server检测和状态保存
        let isLiveServer = false;
        
        function detectLiveServer() {{
            if (window.location.protocol === 'http:' && 
                (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {{
                isLiveServer = true;
                const notice = document.createElement('div');
                notice.className = 'live-server-notice';
                notice.innerHTML = '<strong>💡 Live Server检测</strong><br>已启用状态保存<br><small>点击关闭</small>';
                notice.onclick = () => notice.remove();
                document.body.appendChild(notice);
                setTimeout(() => notice.remove(), 4000);
            }}
        }}
        
        function saveState() {{
            if (!isLiveServer) return;
            const state = {{
                phase: currentPhase,
                methodId: currentMethod ? currentMethod.id : null,
                step: currentStep,
                timestamp: Date.now()
            }};
            localStorage.setItem('attackDemoState', JSON.stringify(state));
        }}
        
        function loadState() {{
            if (!isLiveServer) return null;
            const saved = localStorage.getItem('attackDemoState');
            if (saved) {{
                const state = JSON.parse(saved);
                if (Date.now() - state.timestamp < 5 * 60 * 1000) {{
                    return state;
                }}
            }}
            return null;
        }}
        
        function clearState() {{
            if (isLiveServer) {{
                localStorage.removeItem('attackDemoState');
            }}
        }}
        
        // 方法配置
        const METHODS = {{
            deanonymization: [
                {{
                    id: 'greedy',
                    name: '贪心特征匹配',
                    description: '逐步展示贪心匹配算法的执行过程。每一步选择相似度最高的节点对进行匹配。',
                    principle: `
                        <h4>🔬 算法原理</h4>
                        <div class="formula">
                          步骤1: 计算特征向量<br>
                          f(v) = [degree(v), clustering(v), triangles(v), ...]
                        </div>
                        <div class="formula">
                          步骤2: 计算相似度矩阵<br>
                          S[i][j] = cosine_similarity(f(vi), f(v'j))
                        </div>
                        <div class="formula">
                          步骤3: 贪心选择<br>
                          while 存在未匹配节点:<br>
                          &nbsp;&nbsp;(i*, j*) = argmax S[i][j]<br>
                          &nbsp;&nbsp;匹配 vi* → v'j*<br>
                          &nbsp;&nbsp;删除第i*行和第j*列
                        </div>
                        <ul>
                          <li>每步选择当前最优，不保证全局最优</li>
                          <li>时间复杂度: O(n³)</li>
                          <li>可能陷入局部最优</li>
                        </ul>
                    `
                }}
            ],
            attribute: [
                {{
                    id: 'neighbor_voting',
                    name: '邻居投票',
                    description: '展示如何通过邻居的标签投票来推断未知节点的标签。',
                    principle: `
                        <h4>🔬 算法原理</h4>
                        <div class="formula">
                          对于未知标签的节点v:<br>
                          1. 收集邻居标签: L = {{label(u) | u ∈ N(v)}}
                        </div>
                        <div class="formula">
                          2. 统计标签频率:<br>
                          votes[label] = |{{u ∈ N(v) | label(u) = label}}|
                        </div>
                        <div class="formula">
                          3. 多数投票:<br>
                          predicted_label(v) = argmax votes[label]
                        </div>
                        <ul>
                          <li>基于同质性假设：相连节点倾向于有相同标签</li>
                          <li>简单高效，易于理解</li>
                          <li>对孤立节点效果差</li>
                        </ul>
                    `
                }},
                {{
                    id: 'label_propagation',
                    name: '标签传播',
                    description: '展示标签如何从已知节点逐步传播到未知节点。',
                    principle: `
                        <h4>🔬 算法原理</h4>
                        <div class="formula">
                          初始化:<br>
                          对于v ∈ V_known: label(v) = known_label<br>
                          对于v ∈ V_unknown: label(v) = None
                        </div>
                        <div class="formula">
                          迭代传播 (最多T次):<br>
                          for each v ∈ V_unknown:<br>
                          &nbsp;&nbsp;if label(v) is None:<br>
                          &nbsp;&nbsp;&nbsp;&nbsp;neighbor_labels = [label(u) | u ∈ N(v), label(u) ≠ None]<br>
                          &nbsp;&nbsp;&nbsp;&nbsp;label(v) = most_common(neighbor_labels)
                        </div>
                        <div class="formula">
                          收敛条件:<br>
                          本次迭代无标签更新 或 达到最大迭代次数
                        </div>
                        <ul>
                          <li>标签像"波纹"一样扩散</li>
                          <li>已知标签节点作为"种子"</li>
                          <li>适用于社区结构明显的图</li>
                        </ul>
                    `
                }}
            ],
            robustness: [
                {{
                    id: 'edge_missing',
                    name: '边缺失影响',
                    description: '展示逐步移除边对图结构的影响。红色虚线表示被移除的边。',
                    principle: `
                        <h4>🔬 测试原理</h4>
                        <div class="formula">
                          对于缺失率r ∈ {{0.1, 0.2, 0.3, 0.4, 0.5}}:<br>
                          1. 随机选择 r × |E| 条边<br>
                          2. 从图G中移除这些边 → G_incomplete<br>
                          3. 在G_incomplete上运行攻击<br>
                          4. 测量攻击准确率
                        </div>
                        <div class="formula">
                          观察指标:<br>
                          - 准确率 vs 缺失率曲线<br>
                          - 临界缺失率（攻击失效点）<br>
                          - 图连通性变化
                        </div>
                        <ul>
                          <li>模拟现实中的不完整数据</li>
                          <li>测试攻击的鲁棒性</li>
                          <li>本演示使用贪心匹配攻击</li>
                        </ul>
                    `
                }}
            ],
            defense: [
                {{
                    id: 'differential_privacy',
                    name: '差分隐私防御',
                    description: '展示逐步添加噪声边来保护隐私。绿色边表示添加的噪声。',
                    principle: `
                        <h4>🔬 防御原理</h4>
                        <div class="formula">
                          ε-差分隐私定义:<br>
                          对于任意相差一条边的图G, G':<br>
                          P(M(G) ∈ S) / P(M(G') ∈ S) ≤ e^ε
                        </div>
                        <div class="formula">
                          边扰动机制:<br>
                          1. 删除边: P(删除) = 1/(1 + e^ε)<br>
                          2. 添加边: P(添加) = 1/(1 + e^ε)
                        </div>
                        <div class="formula">
                          隐私预算ε的影响:<br>
                          - ε小（如0.5）: 强隐私保护，大量扰动<br>
                          - ε大（如5.0）: 弱隐私保护，少量扰动
                        </div>
                        <ul>
                          <li>平衡隐私保护和数据效用</li>
                          <li>理论可证明的隐私保证</li>
                          <li>ε越小隐私越强但效用损失越大</li>
                        </ul>
                    `
                }}
            ]
        }};
        
        // 初始化
        function init() {{
            detectLiveServer();
            setupPhaseButtons();
            setupControlButtons();
            initializeCharts();
            
            // 尝试恢复状态
            const savedState = loadState();
            if (savedState) {{
                console.log('恢复状态:', savedState);
                currentPhase = savedState.phase;
                currentStep = savedState.step;
                
                document.querySelectorAll('.phase-btn').forEach(btn => {{
                    btn.classList.toggle('active', btn.dataset.phase === currentPhase);
                }});
                
                updateMethodSelector(currentPhase);
                
                if (savedState.methodId) {{
                    const method = METHODS[currentPhase].find(m => m.id === savedState.methodId);
                    if (method) {{
                        setTimeout(() => {{
                            const methodBtn = document.querySelector(`[data-method-id="${{savedState.methodId}}"]`);
                            if (methodBtn) {{
                                methodBtn.classList.add('active');
                                selectMethod(method, true);
                                for (let i = 0; i < savedState.step; i++) {{
                                    nextStep(true);
                                }}
                            }}
                        }}, 100);
                    }}
                }}
            }} else {{
                updateMethodSelector('deanonymization');
            }}
        }}
        
        function setupPhaseButtons() {{
            document.querySelectorAll('.phase-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    document.querySelectorAll('.phase-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    currentPhase = e.target.dataset.phase;
                    currentStep = 0;
                    updateMethodSelector(currentPhase);
                    resetVisualization();
                    saveState();
                }});
            }});
        }}
        
        function setupControlButtons() {{
            document.getElementById('play-btn').addEventListener('click', playAnimation);
            document.getElementById('next-btn').addEventListener('click', () => nextStep(false));
            document.getElementById('reset-btn').addEventListener('click', resetVisualization);
        }}
        
        function updateMethodSelector(phase) {{
            const container = document.getElementById('method-buttons');
            container.innerHTML = '';
            
            METHODS[phase].forEach((method, idx) => {{
                const btn = document.createElement('button');
                btn.className = 'method-btn' + (idx === 0 ? ' active' : '');
                btn.textContent = method.name;
                btn.dataset.methodId = method.id;
                btn.addEventListener('click', (e) => {{
                    document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    selectMethod(method);
                }});
                container.appendChild(btn);
            }});
            
            if (METHODS[phase].length > 0) {{
                selectMethod(METHODS[phase][0], true);
            }}
        }}
        
        function selectMethod(method, skipSave = false) {{
            currentMethod = method;
            currentStep = 0;
            
            // 显示原理
            const principleBox = document.getElementById('principle');
            if (method.principle) {{
                principleBox.innerHTML = method.principle;
                principleBox.style.display = 'block';
            }} else {{
                principleBox.style.display = 'none';
            }}
            
            // 显示说明
            document.getElementById('explanation').innerHTML = `
                <h4>${{method.name}}</h4>
                <p>${{method.description}}</p>
            `;
            
            resetVisualization();
            prepareVisualization(currentPhase, method.id);
            
            if (!skipSave) saveState();
        }}
        
        function initializeCharts() {{
            topChart = new GraphChart('graph-top', DATA.graphOrig);
            bottomChart = new GraphChart('graph-bottom', DATA.graphAnon);
        }}
        
        function prepareVisualization(phase, methodId) {{
            if (phase === 'deanonymization') {{
                prepareGreedyMatchingViz();
            }} else if (phase === 'attribute') {{
                if (methodId === 'neighbor_voting') {{
                    prepareNeighborVotingViz();
                }} else if (methodId === 'label_propagation') {{
                    prepareLabelPropagationViz();
                }}
            }} else if (phase === 'robustness') {{
                prepareRobustnessViz();
            }} else if (phase === 'defense') {{
                prepareDefenseViz();
            }}
        }}
        
        function prepareGreedyMatchingViz() {{
            document.getElementById('graph-top-title').textContent = '原始图';
            document.getElementById('graph-bottom-title').textContent = '匿名图';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphAnon);
            
            const greedyData = DATA.greedyMatching;
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 贪心匹配步骤</h3>';
            
            greedyData.steps.forEach((step, idx) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.id = `step-${{idx}}`;
                
                const correctIcon = step.is_correct ? '✅' : '❌';
                
                stepDiv.innerHTML = `
                    <strong>步骤 ${{step.step}}:</strong> 匹配节点 <strong>${{step.orig_node}}</strong><br>
                    <div class="detail">
                        原始节点特征:<br>
                        - 度数: ${{step.orig_features.degree}}<br>
                        - 聚类系数: ${{step.orig_features.clustering.toFixed(3)}}<br>
                        <br>
                        最佳匹配: <strong>${{step.matched_node}}</strong><br>
                        相似度: ${{(step.similarity * 100).toFixed(1)}}% ${{correctIcon}}<br>
                        <br>
                        前5个候选:<br>
                        ${{step.candidates.slice(0, 5).map((c, i) => 
                            `${{i+1}}. 节点${{c.node}} (相似度: ${{(c.similarity*100).toFixed(1)}}%)`
                        ).join('<br>')}}
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            }});
            
            const correctCount = greedyData.steps.filter(s => s.is_correct).length;
            updateStats({{
                '总节点数': greedyData.nodes_orig.length,
                '已匹配': greedyData.steps.length,
                '正确匹配': correctCount,
                '准确率': ((correctCount / greedyData.steps.length) * 100).toFixed(0) + '%'
            }});
        }}
        
        function prepareNeighborVotingViz() {{
            if (!DATA.neighborVoting) {{
                document.getElementById('steps-container').innerHTML = 
                    '<p style="color: red;">该数据集不支持属性推断演示</p>';
                return;
            }}
            
            document.getElementById('graph-top-title').textContent = '已知标签节点（彩色）';
            document.getElementById('graph-bottom-title').textContent = '未知标签节点（灰色）';
            
            const graphWithLabels = JSON.parse(JSON.stringify(DATA.graphOrig));
            graphWithLabels.nodes.forEach(node => {{
                const nodeId = node.id;
                if (DATA.neighborVoting.known_labels[nodeId] !== undefined) {{
                    node.label = DATA.neighborVoting.known_labels[nodeId];
                    node.known = true;
                }} else if (DATA.neighborVoting.hidden_labels[nodeId] !== undefined) {{
                    node.label = null;
                    node.known = false;
                }}
            }});
            
            topChart.updateData(graphWithLabels);
            bottomChart.updateData(graphWithLabels);
            
            const votingData = DATA.neighborVoting;
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 邻居投票步骤</h3>';
            
            votingData.steps.forEach((step, idx) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.id = `step-${{idx}}`;
                
                const correctIcon = step.is_correct ? '✅' : '❌';
                const votesBadges = Object.entries(step.votes).map(([label, count]) => 
                    `<span class="vote-badge">标签${{label}}: ${{count}}票</span>`
                ).join('');
                
                stepDiv.innerHTML = `
                    <strong>步骤 ${{step.step}}:</strong> 预测节点 <strong>${{step.target_node}}</strong><br>
                    <div class="detail">
                        邻居数量: ${{step.neighbors.length}}<br>
                        <div class="votes">${{votesBadges}}</div>
                        <br>
                        预测: <strong>标签${{step.prediction}}</strong><br>
                        真实: 标签${{step.actual}} ${{correctIcon}}
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            }});
            
            const correctCount = votingData.steps.filter(s => s.is_correct).length;
            updateStats({{
                '已知标签': Object.keys(DATA.neighborVoting.known_labels).length,
                '未知标签': Object.keys(DATA.neighborVoting.hidden_labels).length,
                '预测节点': votingData.steps.length,
                '正确预测': correctCount
            }});
        }}
        
        function prepareLabelPropagationViz() {{
            if (!DATA.labelPropagation) {{
                document.getElementById('steps-container').innerHTML = 
                    '<p style="color: red;">该数据集不支持标签传播演示</p>';
                return;
            }}
            
            document.getElementById('graph-top-title').textContent = '初始状态';
            document.getElementById('graph-bottom-title').textContent = '标签传播过程';
            
            const graphWithLabels = JSON.parse(JSON.stringify(DATA.graphOrig));
            graphWithLabels.nodes.forEach(node => {{
                const nodeId = node.id;
                if (DATA.labelPropagation.initial_known[nodeId] !== undefined) {{
                    node.label = DATA.labelPropagation.initial_known[nodeId];
                    node.known = true;
                }} else {{
                    node.label = null;
                    node.known = false;
                }}
            }});
            
            topChart.updateData(graphWithLabels);
            bottomChart.updateData(graphWithLabels);
            
            const propData = DATA.labelPropagation;
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 标签传播迭代</h3>';
            
            propData.iterations.forEach((iter, idx) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.id = `step-${{idx}}`;
                
                stepDiv.innerHTML = `
                    <strong>迭代 ${{iter.iteration}}:</strong> 更新了 <strong>${{iter.updated_nodes.length}}</strong> 个节点<br>
                    <div class="detail">
                        ${{iter.updated_nodes.slice(0, 3).map(u => 
                            `节点${{u.node}} → 标签${{u.new_label}} (${{
                                Object.entries(u.votes).map(([l,c]) => `标签${{l}}:${{c}}`).join(', ')
                            }})`
                        ).join('<br>')}}
                        ${{iter.updated_nodes.length > 3 ? '<br>...' : ''}}
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            }});
            
            const totalUpdated = propData.iterations.reduce((sum, iter) => sum + iter.updated_nodes.length, 0);
            updateStats({{
                '迭代次数': propData.iterations.length,
                '初始已知': Object.keys(propData.initial_known).length,
                '传播标注': totalUpdated,
                '覆盖率': ((totalUpdated / propData.initial_unknown.length) * 100).toFixed(0) + '%'
            }});
        }}
        
        function prepareRobustnessViz() {{
            document.getElementById('graph-top-title').textContent = '原始完整图';
            document.getElementById('graph-bottom-title').textContent = '逐步移除边（红色虚线）';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphOrig);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 鲁棒性测试步骤</h3>';
            
            DATA.robustness.forEach((change, idx) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.id = `step-${{idx}}`;
                
                stepDiv.innerHTML = `
                    <strong>缺失率 ${{(change.ratio * 100).toFixed(0)}}%:</strong><br>
                    <div class="detail">
                        本次移除: <span style="color: #ff6b6b;"><strong>${{change.new_removed.length}}</strong></span> 条边<br>
                        累计移除: ${{change.total_removed}} 条<br>
                        剩余边数: ${{change.remaining}} 条
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            }});
            
            updateStats({{
                '原始边数': DATA.graphOrig.links.length,
                '测试阶段': DATA.robustness.length,
                '当前移除': 0,
                '剩余边数': DATA.graphOrig.links.length
            }});
        }}
        
        function prepareDefenseViz() {{
            document.getElementById('graph-top-title').textContent = '原始图';
            document.getElementById('graph-bottom-title').textContent = '逐步添加噪声边（绿色）';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphOrig);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 差分隐私防御步骤</h3>';
            
            DATA.defense.forEach((change, idx) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.id = `step-${{idx}}`;
                
                stepDiv.innerHTML = `
                    <strong>ε = ${{change.epsilon}}:</strong> 隐私强度 <strong>${{change.privacy_level}}</strong><br>
                    <div class="detail">
                        本次添加: <span style="color: #51cf66;"><strong>${{change.new_added.length}}</strong></span> 条噪声边<br>
                        累计添加: ${{change.total_added}} 条<br>
                        总边数: ${{DATA.graphOrig.links.length + change.total_added}} 条
                    </div>
                `;
                stepsContainer.appendChild(stepDiv);
            }});
            
            updateStats({{
                '原始边数': DATA.graphOrig.links.length,
                '防御级别': DATA.defense.length,
                '当前噪声': 0,
                '总边数': DATA.graphOrig.links.length
            }});
        }}
        
        function playAnimation() {{
            if (isPlaying) {{
                stopAnimation();
                return;
            }}
            
            isPlaying = true;
            document.getElementById('play-btn').innerHTML = '⏸️ 暂停';
            
            playInterval = setInterval(() => {{
                if (currentStep >= document.querySelectorAll('.step').length) {{
                    stopAnimation();
                }} else {{
                    nextStep(false);
                }}
            }}, 2000);
        }}
        
        function stopAnimation() {{
            isPlaying = false;
            document.getElementById('play-btn').innerHTML = '▶️ 开始';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}
        
        function nextStep(skipSave) {{
            const steps = document.querySelectorAll('.step');
            if (currentStep < steps.length) {{
                steps.forEach(s => s.classList.remove('current'));
                steps[currentStep].classList.add('current');
                steps[currentStep].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                
                highlightStep(currentStep);
                
                currentStep++;
                if (!skipSave) saveState();
            }}
        }}
        
        function highlightStep(stepIdx) {{
            if (currentPhase === 'deanonymization') {{
                const step = DATA.greedyMatching.steps[stepIdx];
                if (step) {{
                    topChart.highlightNodes([step.orig_node], []);
                    bottomChart.highlightNodes([], step.candidates.map(c => c.node));
                }}
            }} else if (currentPhase === 'attribute') {{
                if (currentMethod.id === 'neighbor_voting') {{
                    const step = DATA.neighborVoting.steps[stepIdx];
                    if (step) {{
                        topChart.highlightNodes([step.target_node], step.neighbors.map(n => n.node));
                        bottomChart.highlightNodes([step.target_node], step.neighbors.map(n => n.node));
                    }}
                }} else if (currentMethod.id === 'label_propagation') {{
                    const iter = DATA.labelPropagation.iterations[stepIdx];
                    if (iter) {{
                        const updatedNodes = iter.updated_nodes.map(u => u.node);
                        topChart.highlightNodes(updatedNodes, []);
                        bottomChart.highlightNodes(updatedNodes, []);
                    }}
                }}
            }} else if (currentPhase === 'robustness') {{
                const change = DATA.robustness[stepIdx];
                if (change) {{
                    bottomChart.removeEdges(change.new_removed);
                    updateStats({{
                        '原始边数': DATA.graphOrig.links.length,
                        '测试阶段': DATA.robustness.length,
                        '当前移除': change.total_removed,
                        '剩余边数': change.remaining
                    }});
                }}
            }} else if (currentPhase === 'defense') {{
                const change = DATA.defense[stepIdx];
                if (change) {{
                    bottomChart.addEdges(change.new_added);
                    updateStats({{
                        '原始边数': DATA.graphOrig.links.length,
                        '防御级别': DATA.defense.length,
                        '当前噪声': change.total_added,
                        '总边数': DATA.graphOrig.links.length + change.total_added
                    }});
                }}
            }}
        }}
        
        function resetVisualization() {{
            stopAnimation();
            currentStep = 0;
            document.querySelectorAll('.step').forEach(s => s.classList.remove('current'));
            
            if (topChart) topChart.resetHighlights();
            if (bottomChart) bottomChart.resetHighlights();
            
            if (currentMethod) {{
                prepareVisualization(currentPhase, currentMethod.id);
            }}
            
            clearState();
        }}
        
        function updateStats(stats) {{
            const container = document.getElementById('stats');
            container.innerHTML = '';
            
            Object.entries(stats).forEach(([label, value]) => {{
                const card = document.createElement('div');
                card.className = 'stat-card';
                card.innerHTML = `
                    <div class="value">${{value}}</div>
                    <div class="label">${{label}}</div>
                `;
                container.appendChild(card);
            }});
        }}
        
        // 图表类
        class GraphChart {{
            constructor(svgId, data) {{
                this.svgId = svgId;
                this.svg = d3.select(`#${{svgId}}`);
                this.width = this.svg.node().clientWidth;
                this.height = this.svg.node().clientHeight;
                
                this.svg.selectAll('*').remove();
                this.g = this.svg.append('g');
                
                const zoom = d3.zoom()
                    .scaleExtent([0.3, 3])
                    .on('zoom', (event) => {{
                        this.g.attr('transform', event.transform);
                    }});
                
                this.svg.call(zoom);
                
                this.simulation = null;
                this.updateData(data);
            }}
            
            updateData(data) {{
                this.data = JSON.parse(JSON.stringify(data));
                this.render();
            }}
            
            render() {{
                this.g.selectAll('*').remove();
                
                this.simulation = d3.forceSimulation(this.data.nodes)
                    .force('link', d3.forceLink(this.data.links)
                        .id(d => d.index)
                        .distance(45))
                    .force('charge', d3.forceManyBody().strength(-130))
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .force('collision', d3.forceCollide().radius(15))
                    .alpha(1)
                    .alphaDecay(0.02);
                
                this.links = this.g.append('g')
                    .selectAll('line')
                    .data(this.data.links)
                    .join('line')
                    .attr('class', 'link');
                
                this.nodes = this.g.append('g')
                    .selectAll('circle')
                    .data(this.data.nodes)
                    .join('circle')
                    .attr('class', 'node')
                    .attr('r', d => 5 + Math.sqrt(d.degree || 1) * 1.3)
                    .attr('fill', d => {{
                        if (d.label !== undefined) {{
                            const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7'];
                            return d.known === false ? '#ddd' : colors[d.label % colors.length];
                        }}
                        return '#4dabf7';
                    }})
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 2)
                    .call(this.drag(this.simulation))
                    .on('mouseover', (event, d) => this.showTooltip(event, d))
                    .on('mouseout', () => this.hideTooltip());
                
                if (this.data.nodes.length < 80) {{
                    this.labels = this.g.append('g')
                        .selectAll('text')
                        .data(this.data.nodes)
                        .join('text')
                        .text(d => d.id)
                        .attr('font-size', 8)
                        .attr('dx', 8)
                        .attr('dy', 3)
                        .style('pointer-events', 'none')
                        .style('opacity', 0.7);
                }}
                
                this.simulation.on('tick', () => {{
                    this.links
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    this.nodes
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                    
                    if (this.labels) {{
                        this.labels
                            .attr('x', d => d.x)
                            .attr('y', d => d.y);
                    }}
                }});
                
                setTimeout(() => this.simulation.stop(), 3000);
            }}
            
            drag(simulation) {{
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
                
                return d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended);
            }}
            
            highlightNodes(primaryNodes, secondaryNodes) {{
                this.resetHighlights();
                
                if (this.nodes) {{
                    this.nodes
                        .classed('highlighted', d => primaryNodes.includes(d.id))
                        .classed('candidate', d => secondaryNodes.includes(d.id))
                        .classed('neighbor', d => secondaryNodes.includes(d.id));
                }}
            }}
            
            resetHighlights() {{
                if (this.nodes) {{
                    this.nodes
                        .classed('highlighted', false)
                        .classed('candidate', false)
                        .classed('neighbor', false)
                        .classed('matched', false);
                }}
                if (this.links) {{
                    this.links
                        .classed('removed', false)
                        .classed('added', false);
                }}
            }}
            
            removeEdges(edgesToRemove) {{
                edgesToRemove.forEach(edge => {{
                    this.links.each(function(d) {{
                        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                        
                        if ((sourceId === edge.source && targetId === edge.target) ||
                            (sourceId === edge.target && targetId === edge.source)) {{
                            d3.select(this).classed('removed', true);
                        }}
                    }});
                }});
            }}
            
            addEdges(edgesToAdd) {{
                edgesToAdd.forEach(edge => {{
                    const sourceNode = this.data.nodes.find(n => n.id === edge.source);
                    const targetNode = this.data.nodes.find(n => n.id === edge.target);
                    
                    if (sourceNode && targetNode) {{
                        const newEdge = {{
                            source: sourceNode.index,
                            target: targetNode.index
                        }};
                        this.data.links.push(newEdge);
                        
                        const newLink = this.g.select('g').append('line')
                            .datum(newEdge)
                            .attr('class', 'link added')
                            .attr('x1', sourceNode.x)
                            .attr('y1', sourceNode.y)
                            .attr('x2', targetNode.x)
                            .attr('y2', targetNode.y);
                        
                        this.simulation.force('link').links(this.data.links);
                        this.simulation.alpha(0.3).restart();
                        setTimeout(() => this.simulation.stop(), 3000);
                    }}
                }});
            }}
            
            showTooltip(event, d) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
                tooltip.innerHTML = `
                    <strong>节点 ${{d.id}}</strong><br>
                    度数: ${{d.degree || 0}}<br>
                    聚类系数: ${{d.clustering ? d.clustering.toFixed(3) : 'N/A'}}
                    ${{d.label !== undefined ? `<br>标签: ${{d.label}}` : ''}}
                `;
            }}
            
            hideTooltip() {{
                document.getElementById('tooltip').style.display = 'none';
            }}
        }}
        
        // 启动
        init();
    </script>
</body>
</html>
"""
        
        return html


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成完整原理演示")
    parser.add_argument('--ego_id', type=str, default='698')
    parser.add_argument('--output', type=str, default='results/attack_principles_complete.html')
    
    args = parser.parse_args()
    
    print("="*70)
    print("完整原理演示系统生成器")
    print("="*70)
    
    visualizer = PrincipleVisualizer(ego_id=args.ego_id)
    output_file = visualizer.generate_html(output_file=args.output)
    
    print("\n" + "="*70)
    print("✅ 生成完成！")
    print(f"📂 文件: {output_file}")
    print("\n特性:")
    print("  ✓ 详细展示算法原理（公式+说明）")
    print("  ✓ 逐步展示执行过程")
    print("  ✓ 增量显示边的变化")
    print("  ✓ Live Server兼容（自动保存状态）")
    print("="*70)


if __name__ == "__main__":
    main()

