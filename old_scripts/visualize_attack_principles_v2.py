"""
改进版交互式攻击原理演示工具
- 修复布局问题
- 增量显示边的变化
- 停止自动刷新
- 优化用户体验
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


class ImprovedAttackVisualizer:
    """改进的攻击原理可视化器"""
    
    def __init__(self, ego_id='698'):
        """初始化可视化器"""
        self.ego_id = ego_id
        print(f"加载 Facebook Ego Network {ego_id}...")
        
        # 加载数据
        loader = DatasetLoader()
        self.G, self.attributes = loader.load_facebook(ego_network=ego_id)
        
        print(f"图规模: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 为可视化选择一个子图（如果网络太大）
        if self.G.number_of_nodes() > 100:
            print("网络较大，选择核心子图用于演示...")
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
        
        # 准备标签数据
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
        """将NetworkX图转换为D3.js格式"""
        nodes = []
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        
        for node in G.nodes():
            node_data = {
                'id': str(node),
                'index': node_to_idx[node],
                'degree': G.degree(node)
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
    
    def prepare_deanonymization_data(self):
        """准备去匿名化攻击的演示数据"""
        print("\n准备去匿名化攻击数据...")
        
        extractor = FeatureExtractor()
        nodes_orig = sorted(list(self.G.nodes()))
        nodes_anon = sorted(list(self.G_anon.nodes()))
        
        features_orig = extractor.extract_node_features(self.G, nodes_orig)
        features_anon = extractor.extract_node_features(self.G_anon, nodes_anon)
        
        scaler = StandardScaler()
        features_orig = scaler.fit_transform(features_orig)
        features_anon = scaler.transform(features_anon)
        
        similarity = cosine_similarity(features_orig, features_anon)
        
        ground_truth = {}
        for orig_node in nodes_orig:
            if orig_node in self.node_mapping:
                anon_node = self.node_mapping[orig_node]
                if anon_node in nodes_anon:
                    ground_truth[str(orig_node)] = str(anon_node)
        
        top_k = 5
        candidates = {}
        for i, orig_node in enumerate(nodes_orig):
            top_indices = np.argsort(similarity[i])[::-1][:top_k]
            candidates[str(orig_node)] = [
                {
                    'node': str(nodes_anon[idx]),
                    'similarity': float(similarity[i][idx]),
                    'rank': rank + 1
                }
                for rank, idx in enumerate(top_indices) if idx < len(nodes_anon)
            ]
        
        node_features = {}
        for i, node in enumerate(nodes_orig):
            node_features[str(node)] = {
                'degree': int(self.G.degree(node)),
                'clustering': float(nx.clustering(self.G, node)),
                'neighbors': len(list(self.G.neighbors(node)))
            }
        
        for i, node in enumerate(nodes_anon):
            node_features[f"anon_{node}"] = {
                'degree': int(self.G_anon.degree(node)),
                'clustering': float(nx.clustering(self.G_anon, node)),
                'neighbors': len(list(self.G_anon.neighbors(node)))
            }
        
        return {
            'ground_truth': ground_truth,
            'candidates': candidates,
            'features': node_features
        }
    
    def prepare_attribute_inference_data(self):
        """准备属性推断攻击的演示数据"""
        print("\n准备属性推断攻击数据...")
        
        if not self.node_labels:
            return None
        
        nodes_list = list(self.node_labels.keys())
        np.random.seed(42)
        nodes_to_hide = set(np.random.choice(nodes_list, len(nodes_list) // 2, replace=False))
        
        known_labels = {str(n): int(self.node_labels[n]) 
                       for n in nodes_list if n not in nodes_to_hide}
        hidden_labels = {str(n): int(self.node_labels[n]) 
                        for n in nodes_to_hide}
        
        neighbor_predictions = {}
        for test_node in nodes_to_hide:
            neighbors = list(self.G.neighbors(test_node))
            neighbor_labels = [self.node_labels[n] for n in neighbors 
                             if n in self.node_labels and n not in nodes_to_hide]
            
            if neighbor_labels:
                label_counts = Counter(neighbor_labels)
                neighbor_predictions[str(test_node)] = {
                    'prediction': int(label_counts.most_common(1)[0][0]),
                    'votes': {int(k): int(v) for k, v in label_counts.items()},
                    'neighbors': [str(n) for n in neighbors if n in self.node_labels and n not in nodes_to_hide]
                }
        
        label_propagation = self._simulate_label_propagation(nodes_to_hide)
        
        return {
            'known_labels': known_labels,
            'hidden_labels': hidden_labels,
            'neighbor_predictions': neighbor_predictions,
            'label_propagation': label_propagation
        }
    
    def _simulate_label_propagation(self, nodes_to_hide, max_iterations=10):
        """模拟标签传播过程"""
        G_copy = self.G.copy()
        
        for node in G_copy.nodes():
            if node not in nodes_to_hide:
                G_copy.nodes[node]['label'] = self.node_labels.get(node)
            else:
                G_copy.nodes[node]['label'] = None
        
        iterations = []
        
        for iteration in range(max_iterations):
            updated_count = 0
            iter_data = {'iteration': iteration + 1, 'updates': []}
            
            for test_node in nodes_to_hide:
                if G_copy.nodes[test_node]['label'] is None:
                    neighbors = list(G_copy.neighbors(test_node))
                    neighbor_labels = [G_copy.nodes[n]['label'] for n in neighbors 
                                     if G_copy.nodes[n]['label'] is not None]
                    
                    if neighbor_labels:
                        label_counts = Counter(neighbor_labels)
                        most_common = label_counts.most_common(1)[0][0]
                        G_copy.nodes[test_node]['label'] = most_common
                        updated_count += 1
                        
                        iter_data['updates'].append({
                            'node': str(test_node),
                            'new_label': int(most_common),
                            'votes': {int(k): int(v) for k, v in label_counts.items()}
                        })
            
            iterations.append(iter_data)
            
            if updated_count == 0:
                break
        
        return iterations
    
    def prepare_robustness_data(self):
        """准备鲁棒性测试数据 - 增量显示"""
        print("\n准备鲁棒性测试数据...")
        
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        edge_changes = {}
        
        all_edges = list(self.G.edges())
        np.random.seed(42)
        
        cumulative_removed = []
        for ratio in missing_ratios:
            num_to_remove = int(len(all_edges) * ratio)
            edges_to_remove_indices = np.random.choice(
                len(all_edges), num_to_remove, replace=False
            )
            
            removed_edges = [
                {'source': str(all_edges[idx][0]), 'target': str(all_edges[idx][1])}
                for idx in edges_to_remove_indices
            ]
            
            # 计算本次新增的移除边
            new_removed = [e for e in removed_edges if e not in cumulative_removed]
            cumulative_removed = removed_edges
            
            edge_changes[f"{ratio:.1f}"] = {
                'removed': new_removed,
                'total_removed': len(removed_edges),
                'remaining': len(all_edges) - len(removed_edges)
            }
        
        return edge_changes
    
    def prepare_defense_data(self):
        """准备防御机制数据 - 增量显示"""
        print("\n准备防御机制数据...")
        
        epsilon_values = [0.5, 1.0, 2.0, 5.0]
        edge_changes = {}
        
        nodes = list(self.G.nodes())
        np.random.seed(42)
        
        cumulative_added = []
        for epsilon in epsilon_values:
            num_noise_edges = int(self.G.number_of_edges() * 0.1 / epsilon)
            
            added_edges = []
            attempts = 0
            while len(added_edges) < num_noise_edges and attempts < num_noise_edges * 10:
                u = np.random.choice(nodes)
                v = np.random.choice(nodes)
                if u != v and not self.G.has_edge(u, v):
                    edge = {'source': str(u), 'target': str(v)}
                    if edge not in added_edges and edge not in cumulative_added:
                        added_edges.append(edge)
                attempts += 1
            
            # 本次新增的边
            new_added = [e for e in added_edges if e not in cumulative_added]
            cumulative_added.extend(new_added)
            
            edge_changes[f"{epsilon:.1f}"] = {
                'added': new_added,
                'total_added': len(cumulative_added),
                'privacy_level': '强' if epsilon < 1 else '中' if epsilon < 2 else '弱'
            }
        
        return edge_changes
    
    def generate_html(self, output_file="results/attack_demo_improved.html"):
        """生成改进的交互式HTML"""
        print("\n生成改进版HTML可视化...")
        
        # 准备所有数据
        graph_orig = self.graph_to_json(self.G, include_labels=True)
        graph_anon = self.graph_to_json(self.G_anon)
        deanon_data = self.prepare_deanonymization_data()
        attr_data = self.prepare_attribute_inference_data()
        robust_data = self.prepare_robustness_data()
        defense_data = self.prepare_defense_data()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        html_content = self._generate_html_template(
            graph_orig, graph_anon, deanon_data, attr_data, 
            robust_data, defense_data
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML已生成: {output_file}")
        return output_file
    
    def _generate_html_template(self, graph_orig, graph_anon, deanon_data, 
                                attr_data, robust_data, defense_data):
        """生成HTML模板"""
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图去匿名化攻击原理演示 - 改进版</title>
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
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 1.8em;
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
            min-height: 400px;
        }}
        
        .graph-container h3 {{
            margin-bottom: 10px;
            color: #495057;
            font-size: 1.1em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        
        .graph-svg {{
            width: 100%;
            height: calc(100% - 50px);
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
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .phase-selector h2 {{
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .phase-buttons {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .phase-btn {{
            padding: 12px 20px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            background: white;
            color: #495057;
            font-size: 1em;
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
            padding: 15px 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .method-selector h3 {{
            font-size: 1em;
            margin-bottom: 10px;
            color: #6c757d;
        }}
        
        .method-buttons {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .method-btn {{
            padding: 10px 15px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: white;
            color: #495057;
            font-size: 0.95em;
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
            padding: 20px;
            overflow-y: auto;
        }}
        
        .explanation {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #ffc107;
        }}
        
        .explanation h4 {{
            margin-bottom: 8px;
            color: #856404;
        }}
        
        .explanation p {{
            color: #856404;
            line-height: 1.6;
            font-size: 0.95em;
        }}
        
        .steps-container {{
            margin-top: 15px;
        }}
        
        .steps-container h3 {{
            font-size: 1.05em;
            margin-bottom: 12px;
            color: #495057;
        }}
        
        .step {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            margin: 8px 0;
            border-left: 3px solid #28a745;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }}
        
        .step.current {{
            background: #e7f3ff;
            border-left-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }}
        
        .step strong {{
            color: #495057;
        }}
        
        .controls {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 2px solid #e9ecef;
            display: flex;
            gap: 10px;
        }}
        
        .control-btn {{
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 0.95em;
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
            gap: 10px;
            margin-top: 15px;
        }}
        
        .stat-card {{
            background: white;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            font-size: 0.85em;
            margin-top: 4px;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
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
            stroke-opacity: 0.3;
            stroke-dasharray: 5,5;
            animation: dash 1s linear;
        }}
        
        .link.added {{
            stroke: #51cf66;
            stroke-opacity: 0.8;
            stroke-width: 2;
            animation: pulse 1s ease-in-out;
        }}
        
        @keyframes dash {{
            from {{ stroke-dashoffset: 0; }}
            to {{ stroke-dashoffset: 100; }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ stroke-opacity: 0.4; }}
            50% {{ stroke-opacity: 1; }}
        }}
        
        .node.highlighted {{
            stroke: #ff6b6b;
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
            padding: 10px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            font-size: 0.9em;
            z-index: 1000;
            display: none;
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <header>
            <h1>🔍 图去匿名化攻击原理演示系统 v2.0</h1>
            <p>交互式可视化 - 深入理解攻击和防御机制 | 数据集: Facebook Ego {self.ego_id} ({self.G.number_of_nodes()}节点, {self.G.number_of_edges()}边)</p>
        </header>
        
        <div class="content-wrapper">
            <!-- 左侧：图可视化区域 -->
            <div class="graphs-panel">
                <div class="graph-container">
                    <h3 id="graph-top-title">原始图</h3>
                    <svg id="graph-top" class="graph-svg"></svg>
                </div>
                
                <div class="graph-container">
                    <h3 id="graph-bottom-title">匿名图/修改后的图</h3>
                    <svg id="graph-bottom" class="graph-svg"></svg>
                </div>
            </div>
            
            <!-- 右侧：控制面板 -->
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
                    <div id="explanation" class="explanation">
                        <h4>选择一个方法开始演示</h4>
                        <p>请在上方选择要演示的攻击阶段和具体方法...</p>
                    </div>
                    
                    <div id="steps-container" class="steps-container"></div>
                    
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #4dabf7;"></div>
                            <span>普通节点</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff6b6b;"></div>
                            <span>当前选中</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #51cf66;"></div>
                            <span>匹配成功</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffd43b;"></div>
                            <span>候选节点</span>
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
            graphOrig: {json.dumps(graph_orig)},
            graphAnon: {json.dumps(graph_anon)},
            deanonymization: {json.dumps(deanon_data)},
            attribute: {json.dumps(attr_data)},
            robustness: {json.dumps(robust_data)},
            defense: {json.dumps(defense_data)}
        }};
        
        // 全局状态
        let currentPhase = 'deanonymization';
        let currentMethod = null;
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        
        // D3 图表实例
        let topChart = null;
        let bottomChart = null;
        
        // 方法配置
        const METHODS = {{
            deanonymization: [
                {{
                    id: 'greedy',
                    name: '贪心特征匹配',
                    description: '基于节点结构特征的贪心匹配算法。计算每个节点的度数、聚类系数等特征，然后在原始图和匿名图之间找到特征最相似的节点配对。'
                }},
                {{
                    id: 'hungarian',
                    name: '匈牙利算法',
                    description: '使用匈牙利算法求解最优匹配问题。将节点匹配建模为二分图最大权重匹配，通过特征相似度矩阵找到全局最优解。'
                }},
                {{
                    id: 'graphkernel',
                    name: '图核方法',
                    description: '基于图核相似度的匹配方法。考虑节点的局部子图结构，包括1-hop和2-hop邻居信息，计算更丰富的结构相似度。'
                }},
                {{
                    id: 'deepwalk',
                    name: 'DeepWalk嵌入',
                    description: 'DeepWalk图嵌入方法。通过随机游走生成节点序列，使用Word2Vec学习节点的低维向量表示，然后在嵌入空间中进行匹配。'
                }}
            ],
            attribute: [
                {{
                    id: 'neighbor_voting',
                    name: '邻居投票',
                    description: '基于邻居标签的简单投票机制。收集目标节点所有已知标签的邻居，统计标签频率，选择出现次数最多的标签作为预测结果。'
                }},
                {{
                    id: 'label_propagation',
                    name: '标签传播',
                    description: '迭代式标签传播算法。从已知标签节点开始，逐步将标签传播到未知节点，每次迭代中节点采用邻居中最常见的标签。'
                }},
                {{
                    id: 'graphsage',
                    name: 'GraphSAGE',
                    description: 'GraphSAGE图神经网络。通过聚合邻居特征学习节点表示，使用神经网络进行端到端的标签预测训练。'
                }}
            ],
            robustness: [
                {{
                    id: 'missing_edges',
                    name: '边缺失影响（增量显示）',
                    description: '测试在不同边缺失率下攻击效果的变化。每一步增加缺失的边，红色虚线表示被移除的边，观察图结构的逐步退化。'
                }}
            ],
            defense: [
                {{
                    id: 'differential_privacy',
                    name: '差分隐私（增量显示）',
                    description: '通过逐步添加噪声边来保护隐私。绿色高亮边表示新添加的噪声边，ε参数控制隐私保护强度，ε越小添加的噪声越多。'
                }}
            ]
        }};
        
        // 初始化
        function init() {{
            setupPhaseButtons();
            setupControlButtons();
            initializeCharts();
            updateMethodSelector('deanonymization');
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
                }});
            }});
        }}
        
        function setupControlButtons() {{
            document.getElementById('play-btn').addEventListener('click', playAnimation);
            document.getElementById('next-btn').addEventListener('click', nextStep);
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
                selectMethod(METHODS[phase][0]);
            }}
        }}
        
        function selectMethod(method) {{
            currentMethod = method;
            currentStep = 0;
            
            document.getElementById('explanation').innerHTML = `
                <h4>${{method.name}}</h4>
                <p>${{method.description}}</p>
            `;
            
            resetVisualization();
            prepareVisualization(currentPhase, method.id);
        }}
        
        function initializeCharts() {{
            // 创建图表时禁用自动刷新
            topChart = new GraphChart('graph-top', DATA.graphOrig, false);
            bottomChart = new GraphChart('graph-bottom', DATA.graphAnon, false);
        }}
        
        function prepareVisualization(phase, methodId) {{
            if (phase === 'deanonymization') {{
                prepareDeanonymizationViz(methodId);
            }} else if (phase === 'attribute') {{
                prepareAttributeViz(methodId);
            }} else if (phase === 'robustness') {{
                prepareRobustnessViz(methodId);
            }} else if (phase === 'defense') {{
                prepareDefenseViz(methodId);
            }}
        }}
        
        function prepareDeanonymizationViz(methodId) {{
            document.getElementById('graph-top-title').textContent = '原始图';
            document.getElementById('graph-bottom-title').textContent = '匿名图';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphAnon);
            
            const candidates = DATA.deanonymization.candidates;
            const groundTruth = DATA.deanonymization.ground_truth;
            const origNodes = Object.keys(candidates);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 演示步骤</h3>';
            
            const demoNodes = origNodes.slice(0, 5);
            demoNodes.forEach((node, idx) => {{
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                
                const topCandidate = candidates[node][0];
                const isCorrect = groundTruth[node] === topCandidate.node;
                
                step.innerHTML = `
                    <strong>步骤 ${{idx + 1}}:</strong> 
                    匹配节点 <strong>${{node}}</strong><br>
                    特征: 度=${{DATA.deanonymization.features[node].degree}}, 
                    聚类系数=${{DATA.deanonymization.features[node].clustering.toFixed(3)}}<br>
                    最佳匹配: <strong>${{topCandidate.node}}</strong> 
                    (相似度: ${{(topCandidate.similarity * 100).toFixed(1)}}%)
                    ${{isCorrect ? ' ✅' : ' ❌'}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '节点总数': DATA.graphOrig.nodes.length,
                '匹配对数': demoNodes.length,
                '正确匹配': demoNodes.filter(n => groundTruth[n] === candidates[n][0].node).length,
                '准确率': (demoNodes.filter(n => groundTruth[n] === candidates[n][0].node).length / demoNodes.length * 100).toFixed(0) + '%'
            }});
        }}
        
        function prepareAttributeViz(methodId) {{
            if (!DATA.attribute) {{
                document.getElementById('explanation').innerHTML += 
                    '<p style="color: red; margin-top: 10px;">⚠️ 该数据集不支持属性推断演示</p>';
                return;
            }}
            
            document.getElementById('graph-top-title').textContent = '已知标签节点（彩色）';
            document.getElementById('graph-bottom-title').textContent = '标签传播过程';
            
            const graphWithLabels = JSON.parse(JSON.stringify(DATA.graphOrig));
            graphWithLabels.nodes.forEach(node => {{
                if (DATA.attribute.known_labels[node.id]) {{
                    node.label = DATA.attribute.known_labels[node.id];
                    node.known = true;
                }} else if (DATA.attribute.hidden_labels[node.id]) {{
                    node.label = null;
                    node.known = false;
                }}
            }});
            
            topChart.updateData(graphWithLabels);
            bottomChart.updateData(graphWithLabels);
            
            if (methodId === 'neighbor_voting') {{
                prepareNeighborVotingSteps();
            }} else if (methodId === 'label_propagation') {{
                prepareLabelPropagationSteps();
            }} else {{
                prepareGraphSAGESteps();
            }}
        }}
        
        function prepareNeighborVotingSteps() {{
            const predictions = DATA.attribute.neighbor_predictions;
            const hiddenLabels = DATA.attribute.hidden_labels;
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 邻居投票步骤</h3>';
            
            const nodes = Object.keys(predictions).slice(0, 5);
            nodes.forEach((node, idx) => {{
                const pred = predictions[node];
                const actual = hiddenLabels[node];
                const isCorrect = pred.prediction === actual;
                
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>步骤 ${{idx + 1}}:</strong> 
                    预测节点 <strong>${{node}}</strong><br>
                    邻居投票: ${{Object.entries(pred.votes).map(([l, c]) => `标签${{l}}(${{c}}票)`).join(', ')}}<br>
                    预测: 标签${{pred.prediction}} 
                    ${{isCorrect ? '✅ 正确' : '❌ 错误 (真实: 标签' + actual + ')'}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '已知标签': Object.keys(DATA.attribute.known_labels).length,
                '未知标签': Object.keys(DATA.attribute.hidden_labels).length,
                '预测节点': nodes.length,
                '正确预测': nodes.filter(n => predictions[n].prediction === hiddenLabels[n]).length
            }});
        }}
        
        function prepareLabelPropagationSteps() {{
            const iterations = DATA.attribute.label_propagation;
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 标签传播迭代</h3>';
            
            iterations.forEach((iter, idx) => {{
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>迭代 ${{iter.iteration}}:</strong>
                    更新 <strong>${{iter.updates.length}}</strong> 个节点<br>
                    ${{iter.updates.slice(0, 2).map(u => 
                        `节点${{u.node}} → 标签${{u.new_label}}`
                    ).join(', ')}}
                    ${{iter.updates.length > 2 ? '...' : ''}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            const totalUpdated = iterations.reduce((sum, iter) => sum + iter.updates.length, 0);
            updateStats({{
                '迭代次数': iterations.length,
                '初始已知': Object.keys(DATA.attribute.known_labels).length,
                '新标注节点': totalUpdated,
                '覆盖率': ((totalUpdated / Object.keys(DATA.attribute.hidden_labels).length) * 100).toFixed(0) + '%'
            }});
        }}
        
        function prepareGraphSAGESteps() {{
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 GraphSAGE过程</h3>';
            stepsContainer.innerHTML += '<p style="color: #6c757d; padding: 10px;">GraphSAGE使用神经网络进行训练，这里展示概念性流程。</p>';
        }}
        
        function prepareRobustnessViz(methodId) {{
            document.getElementById('graph-top-title').textContent = '原始完整图';
            document.getElementById('graph-bottom-title').textContent = '逐步移除边（红色虚线）';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphOrig); // 从完整图开始
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 鲁棒性测试 - 增量显示</h3>';
            
            const ratios = Object.keys(DATA.robustness);
            ratios.forEach((ratio, idx) => {{
                const change = DATA.robustness[ratio];
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>缺失率 ${{(parseFloat(ratio) * 100).toFixed(0)}}%:</strong><br>
                    本次移除 <strong style="color: #ff6b6b;">${{change.removed.length}}</strong> 条边<br>
                    累计移除 ${{change.total_removed}} 条，剩余 ${{change.remaining}} 条
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '原始边数': DATA.graphOrig.links.length,
                '测试阶段': ratios.length,
                '当前移除': 0,
                '剩余边数': DATA.graphOrig.links.length
            }});
        }}
        
        function prepareDefenseViz(methodId) {{
            document.getElementById('graph-top-title').textContent = '原始图';
            document.getElementById('graph-bottom-title').textContent = '逐步添加噪声边（绿色高亮）';
            
            topChart.updateData(DATA.graphOrig);
            bottomChart.updateData(DATA.graphOrig); // 从原始图开始
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 差分隐私防御 - 增量显示</h3>';
            
            const epsilons = Object.keys(DATA.defense);
            epsilons.forEach((epsilon, idx) => {{
                const change = DATA.defense[epsilon];
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>ε = ${{epsilon}}:</strong> 
                    隐私强度 <span style="color: #667eea; font-weight: bold;">${{change.privacy_level}}</span><br>
                    本次添加 <strong style="color: #51cf66;">${{change.added.length}}</strong> 条噪声边<br>
                    累计添加 ${{change.total_added}} 条噪声边
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '原始边数': DATA.graphOrig.links.length,
                '防御级别': epsilons.length,
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
                    nextStep();
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
        
        function nextStep() {{
            const steps = document.querySelectorAll('.step');
            if (currentStep < steps.length) {{
                steps.forEach(s => s.classList.remove('current'));
                steps[currentStep].classList.add('current');
                steps[currentStep].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                
                highlightStep(currentStep);
                
                currentStep++;
            }}
        }}
        
        function highlightStep(stepIdx) {{
            if (currentPhase === 'deanonymization') {{
                highlightDeanonymizationStep(stepIdx);
            }} else if (currentPhase === 'attribute') {{
                highlightAttributeStep(stepIdx);
            }} else if (currentPhase === 'robustness') {{
                highlightRobustnessStep(stepIdx);
            }} else if (currentPhase === 'defense') {{
                highlightDefenseStep(stepIdx);
            }}
        }}
        
        function highlightDeanonymizationStep(stepIdx) {{
            const candidates = DATA.deanonymization.candidates;
            const origNodes = Object.keys(candidates).slice(0, 5);
            
            if (stepIdx < origNodes.length) {{
                const origNode = origNodes[stepIdx];
                const topCandidates = candidates[origNode].map(c => c.node);
                
                topChart.highlightNodes([origNode]);
                bottomChart.highlightNodes(topCandidates);
            }}
        }}
        
        function highlightAttributeStep(stepIdx) {{
            if (!DATA.attribute) return;
            
            if (currentMethod.id === 'neighbor_voting') {{
                const nodes = Object.keys(DATA.attribute.neighbor_predictions).slice(0, 5);
                if (stepIdx < nodes.length) {{
                    const node = nodes[stepIdx];
                    const neighbors = DATA.attribute.neighbor_predictions[node].neighbors;
                    topChart.highlightNodes([node]);
                    bottomChart.highlightNodes([node]);
                }}
            }} else if (currentMethod.id === 'label_propagation') {{
                const iterations = DATA.attribute.label_propagation;
                if (stepIdx < iterations.length) {{
                    const updates = iterations[stepIdx].updates;
                    const updatedNodes = updates.map(u => u.node);
                    topChart.highlightNodes(updatedNodes);
                    bottomChart.highlightNodes(updatedNodes);
                }}
            }}
        }}
        
        function highlightRobustnessStep(stepIdx) {{
            const ratios = Object.keys(DATA.robustness);
            if (stepIdx < ratios.length) {{
                const ratio = ratios[stepIdx];
                const change = DATA.robustness[ratio];
                
                // 增量添加移除的边
                bottomChart.removeEdges(change.removed);
                
                // 更新统计
                updateStats({{
                    '原始边数': DATA.graphOrig.links.length,
                    '测试阶段': ratios.length,
                    '当前移除': change.total_removed,
                    '剩余边数': change.remaining
                }});
            }}
        }}
        
        function highlightDefenseStep(stepIdx) {{
            const epsilons = Object.keys(DATA.defense);
            if (stepIdx < epsilons.length) {{
                const epsilon = epsilons[stepIdx];
                const change = DATA.defense[epsilon];
                
                // 增量添加噪声边
                bottomChart.addEdges(change.added);
                
                // 更新统计
                updateStats({{
                    '原始边数': DATA.graphOrig.links.length,
                    '防御级别': epsilons.length,
                    '当前噪声': change.total_added,
                    '总边数': DATA.graphOrig.links.length + change.total_added
                }});
            }}
        }}
        
        function resetVisualization() {{
            stopAnimation();
            currentStep = 0;
            document.querySelectorAll('.step').forEach(s => s.classList.remove('current'));
            
            if (topChart) topChart.resetHighlights();
            if (bottomChart) bottomChart.resetHighlights();
            
            // 重新准备可视化
            if (currentMethod) {{
                prepareVisualization(currentPhase, currentMethod.id);
            }}
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
        
        // 图表类 - 优化版，停止自动刷新
        class GraphChart {{
            constructor(svgId, data, autoRefresh = false) {{
                this.svgId = svgId;
                this.svg = d3.select(`#${{svgId}}`);
                this.width = this.svg.node().clientWidth;
                this.height = this.svg.node().clientHeight;
                this.autoRefresh = autoRefresh;
                
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
                
                // 创建力导向布局，但限制迭代次数
                this.simulation = d3.forceSimulation(this.data.nodes)
                    .force('link', d3.forceLink(this.data.links)
                        .id(d => d.index)
                        .distance(50))
                    .force('charge', d3.forceManyBody().strength(-150))
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .force('collision', d3.forceCollide().radius(15))
                    .alpha(1)
                    .alphaDecay(0.02); // 加快衰减，更快停止
                
                // 绘制边
                this.links = this.g.append('g')
                    .selectAll('line')
                    .data(this.data.links)
                    .join('line')
                    .attr('class', 'link');
                
                // 绘制节点
                this.nodes = this.g.append('g')
                    .selectAll('circle')
                    .data(this.data.nodes)
                    .join('circle')
                    .attr('class', 'node')
                    .attr('r', d => 5 + Math.sqrt(d.degree || 1) * 1.5)
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
                
                // 添加标签（仅小图显示）
                if (this.data.nodes.length < 100) {{
                    this.labels = this.g.append('g')
                        .selectAll('text')
                        .data(this.data.nodes)
                        .join('text')
                        .text(d => d.id)
                        .attr('font-size', 9)
                        .attr('dx', 10)
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
                
                // 运行固定次数后停止
                if (!this.autoRefresh) {{
                    setTimeout(() => {{
                        this.simulation.stop();
                    }}, 3000); // 3秒后停止布局计算
                }}
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
            
            highlightNodes(nodeIds) {{
                this.resetHighlights();
                
                this.nodes
                    .classed('highlighted', d => nodeIds.includes(d.id));
            }}
            
            resetHighlights() {{
                if (this.nodes) {{
                    this.nodes
                        .classed('highlighted', false)
                        .classed('candidate', false)
                        .classed('matched', false);
                }}
                if (this.links) {{
                    this.links
                        .classed('removed', false)
                        .classed('added', false);
                }}
            }}
            
            // 增量移除边
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
            
            // 增量添加边
            addEdges(edgesToAdd) {{
                edgesToAdd.forEach(edge => {{
                    // 找到对应的节点对象
                    const sourceNode = this.data.nodes.find(n => n.id === edge.source);
                    const targetNode = this.data.nodes.find(n => n.id === edge.target);
                    
                    if (sourceNode && targetNode) {{
                        // 添加新边到数据
                        const newEdge = {{
                            source: sourceNode.index,
                            target: targetNode.index
                        }};
                        this.data.links.push(newEdge);
                        
                        // 添加到可视化
                        const newLink = this.g.select('g').append('line')
                            .datum(newEdge)
                            .attr('class', 'link added')
                            .attr('x1', sourceNode.x)
                            .attr('y1', sourceNode.y)
                            .attr('x2', targetNode.x)
                            .attr('y2', targetNode.y);
                        
                        // 使用模拟器更新位置
                        this.simulation.force('link').links(this.data.links);
                        this.simulation.alpha(0.3).restart();
                        
                        // 3秒后停止
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
                    度数: ${{d.degree || 0}}
                    ${{d.label !== undefined ? `<br>标签: ${{d.label}}` : ''}}
                `;
            }}
            
            hideTooltip() {{
                document.getElementById('tooltip').style.display = 'none';
            }}
        }}
        
        // 启动应用
        init();
    </script>
</body>
</html>
"""
        
        return html


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成改进版攻击原理演示")
    parser.add_argument('--ego_id', type=str, default='698',
                       help='Ego网络ID')
    parser.add_argument('--output', type=str, default='results/attack_demo_improved.html',
                       help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    print("="*70)
    print("改进版攻击原理可视化演示工具")
    print("="*70)
    
    visualizer = ImprovedAttackVisualizer(ego_id=args.ego_id)
    output_file = visualizer.generate_html(output_file=args.output)
    
    print("\n" + "="*70)
    print("✅ 生成完成！")
    print(f"📂 文件位置: {output_file}")
    print("\n主要改进:")
    print("  ✓ 左右分栏布局（左侧上下两图，右侧控制面板）")
    print("  ✓ 停止自动刷新（3秒后自动停止布局计算）")
    print("  ✓ 增量显示边的变化（鲁棒性和防御）")
    print("  ✓ 优化用户交互体验")
    print("\n使用说明:")
    print("  1. 在浏览器中打开HTML文件")
    print("  2. 选择攻击阶段和方法")
    print("  3. 点击'开始'观看演示")
    print("  4. 鲁棒性/防御阶段会增量显示边的变化")
    print("="*70)


if __name__ == "__main__":
    main()









