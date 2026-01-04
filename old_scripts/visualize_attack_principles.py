"""
交互式攻击原理演示工具
生成可视化HTML，展示各种攻击方法的工作原理
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


class AttackPrincipleVisualizer:
    """攻击原理可视化器"""
    
    def __init__(self, ego_id='698'):
        """
        初始化可视化器
        
        Args:
            ego_id: 使用小规模的ego网络便于可视化
        """
        self.ego_id = ego_id
        print(f"加载 Facebook Ego Network {ego_id}...")
        
        # 加载数据
        loader = DatasetLoader()
        self.G, self.attributes = loader.load_facebook(ego_network=ego_id)
        
        print(f"图规模: {self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边")
        
        # 为可视化选择一个子图（如果网络太大）
        if self.G.number_of_nodes() > 100:
            print("网络较大，选择核心子图用于演示...")
            # 选择度数最高的50个节点
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
        
        # 准备标签数据（用于属性推断）
        self.prepare_labels()
        
    def prepare_labels(self):
        """准备标签数据"""
        self.node_labels = {}
        circle_to_int = {}  # 映射circle字符串到整数
        next_label = 0
        
        for node in self.G.nodes():
            if node in self.attributes and 'circles' in self.attributes[node]:
                circles = self.attributes[node]['circles']
                if circles:
                    circle = circles[0]
                    # 将circle字符串映射为整数
                    if circle not in circle_to_int:
                        circle_to_int[circle] = next_label
                        next_label += 1
                    self.node_labels[node] = circle_to_int[circle]
        
        if not self.node_labels:
            # 如果没有标签，使用度数分组作为伪标签
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
        
        # 计算特征
        extractor = FeatureExtractor()
        nodes_orig = sorted(list(self.G.nodes()))
        nodes_anon = sorted(list(self.G_anon.nodes()))
        
        features_orig = extractor.extract_node_features(self.G, nodes_orig)
        features_anon = extractor.extract_node_features(self.G_anon, nodes_anon)
        
        scaler = StandardScaler()
        features_orig = scaler.fit_transform(features_orig)
        features_anon = scaler.transform(features_anon)
        
        similarity = cosine_similarity(features_orig, features_anon)
        
        # 构建Ground Truth
        ground_truth = {}
        for orig_node in nodes_orig:
            if orig_node in self.node_mapping:
                anon_node = self.node_mapping[orig_node]
                if anon_node in nodes_anon:
                    ground_truth[str(orig_node)] = str(anon_node)
        
        # 为每个原始节点找Top-K候选
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
        
        # 提取特征值用于展示
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
            'features': node_features,
            'similarity_matrix': similarity.tolist()
        }
    
    def prepare_attribute_inference_data(self):
        """准备属性推断攻击的演示数据"""
        print("\n准备属性推断攻击数据...")
        
        if not self.node_labels:
            return None
        
        # 隐藏50%节点的标签
        nodes_list = list(self.node_labels.keys())
        np.random.seed(42)
        nodes_to_hide = set(np.random.choice(nodes_list, len(nodes_list) // 2, replace=False))
        
        known_labels = {str(n): int(self.node_labels[n]) 
                       for n in nodes_list if n not in nodes_to_hide}
        hidden_labels = {str(n): int(self.node_labels[n]) 
                        for n in nodes_to_hide}
        
        # 邻居投票预测
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
        
        # 标签传播
        label_propagation = self._simulate_label_propagation(nodes_to_hide)
        
        return {
            'known_labels': known_labels,
            'hidden_labels': hidden_labels,
            'neighbor_predictions': neighbor_predictions,
            'label_propagation': label_propagation,
            'label_names': self._get_label_names()
        }
    
    def _simulate_label_propagation(self, nodes_to_hide, max_iterations=10):
        """模拟标签传播过程"""
        G_copy = self.G.copy()
        
        # 初始化标签
        for node in G_copy.nodes():
            if node not in nodes_to_hide:
                G_copy.nodes[node]['label'] = self.node_labels.get(node)
            else:
                G_copy.nodes[node]['label'] = None
        
        # 记录每次迭代的状态
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
    
    def _get_label_names(self):
        """获取标签名称"""
        unique_labels = sorted(set(self.node_labels.values()))
        return {int(label): f"Group {label}" for label in unique_labels}
    
    def prepare_robustness_data(self):
        """准备鲁棒性测试数据"""
        print("\n准备鲁棒性测试数据...")
        
        # 生成不同缺失率的图
        missing_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        graphs = {}
        
        for ratio in missing_ratios:
            if ratio == 0:
                G_incomplete = self.G.copy()
            else:
                # 随机移除边
                edges = list(self.G.edges())
                np.random.seed(int(ratio * 100))
                edges_to_remove = np.random.choice(len(edges), 
                                                   int(len(edges) * ratio), 
                                                   replace=False)
                G_incomplete = self.G.copy()
                for idx in edges_to_remove:
                    G_incomplete.remove_edge(*edges[idx])
            
            graphs[f"{ratio:.1f}"] = self.graph_to_json(G_incomplete)
        
        return graphs
    
    def prepare_defense_data(self):
        """准备防御机制数据"""
        print("\n准备防御机制数据...")
        
        epsilon_values = [0.5, 1.0, 2.0, 5.0]
        defended_graphs = {}
        
        for epsilon in epsilon_values:
            # 简单模拟差分隐私（边扰动）
            G_defended = self.G.copy()
            
            # 添加噪声边
            num_noise_edges = int(self.G.number_of_edges() * 0.1 / epsilon)
            nodes = list(G_defended.nodes())
            
            for _ in range(num_noise_edges):
                u = np.random.choice(nodes)
                v = np.random.choice(nodes)
                if u != v and not G_defended.has_edge(u, v):
                    G_defended.add_edge(u, v)
            
            defended_graphs[f"{epsilon:.1f}"] = self.graph_to_json(G_defended)
        
        return defended_graphs
    
    def generate_html(self, output_file="results/attack_principles_demo.html"):
        """生成交互式HTML"""
        print("\n生成HTML可视化...")
        
        # 准备所有数据
        graph_orig = self.graph_to_json(self.G, include_labels=True)
        graph_anon = self.graph_to_json(self.G_anon)
        deanon_data = self.prepare_deanonymization_data()
        attr_data = self.prepare_attribute_inference_data()
        robust_data = self.prepare_robustness_data()
        defense_data = self.prepare_defense_data()
        
        # 确保输出目录存在
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
    <title>图去匿名化攻击原理演示</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        
        .control-panel {{
            background: #f8f9fa;
            padding: 25px;
            border-bottom: 3px solid #e9ecef;
        }}
        
        .phase-selector {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .phase-btn {{
            flex: 1;
            min-width: 200px;
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            color: #495057;
            border: 2px solid #dee2e6;
        }}
        
        .phase-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .phase-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }}
        
        .method-selector {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        
        .method-btn {{
            padding: 10px 20px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            background: white;
            color: #495057;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .method-btn:hover {{
            background: #f8f9fa;
            border-color: #667eea;
        }}
        
        .method-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .visualization-area {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 25px;
        }}
        
        .graph-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .graph-container h3 {{
            margin-bottom: 15px;
            color: #495057;
            font-size: 1.3em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .graph-svg {{
            width: 100%;
            height: 600px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            background: #f8f9fa;
        }}
        
        .info-panel {{
            grid-column: 1 / -1;
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-top: 10px;
        }}
        
        .info-panel h3 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .explanation {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid #667eea;
        }}
        
        .step {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
        }}
        
        .step.current {{
            background: #e7f3ff;
            border-left-color: #667eea;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ background: #e7f3ff; }}
            50% {{ background: #cfe4ff; }}
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }}
        
        .control-btn {{
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .control-btn.play {{
            background: #28a745;
            color: white;
        }}
        
        .control-btn.play:hover {{
            background: #218838;
            transform: scale(1.05);
        }}
        
        .control-btn.reset {{
            background: #6c757d;
            color: white;
        }}
        
        .control-btn.reset:hover {{
            background: #5a6268;
        }}
        
        .control-btn.next {{
            background: #007bff;
            color: white;
        }}
        
        .control-btn.next:hover {{
            background: #0056b3;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #333;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .node:hover {{
            stroke-width: 4px;
        }}
        
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        
        .node.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 4px;
        }}
        
        .node.matched {{
            stroke: #51cf66;
            stroke-width: 4px;
        }}
        
        .node.candidate {{
            stroke: #ffd43b;
            stroke-width: 3px;
        }}
        
        .link.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 3px;
            stroke-opacity: 1;
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
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 图去匿名化攻击原理演示系统</h1>
            <p>交互式可视化 - 深入理解攻击和防御机制</p>
            <p style="font-size: 0.9em; margin-top: 10px;">数据集: Facebook Ego Network {self.ego_id} 
               ({self.G.number_of_nodes()} 节点, {self.G.number_of_edges()} 边)</p>
        </header>
        
        <div class="control-panel">
            <div class="phase-selector">
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
            
            <div id="method-selector" class="method-selector">
                <!-- 动态填充 -->
            </div>
        </div>
        
        <div class="visualization-area">
            <div class="graph-container">
                <h3 id="graph-left-title">原始图</h3>
                <svg id="graph-left" class="graph-svg"></svg>
            </div>
            
            <div class="graph-container">
                <h3 id="graph-right-title">匿名图</h3>
                <svg id="graph-right" class="graph-svg"></svg>
            </div>
            
            <div class="info-panel">
                <h3>📖 算法说明</h3>
                <div id="explanation" class="explanation">
                    选择一个攻击方法开始演示...
                </div>
                
                <div id="steps-container"></div>
                
                <div class="controls">
                    <button class="control-btn play" id="play-btn">▶️ 开始演示</button>
                    <button class="control-btn next" id="next-btn">⏭️ 下一步</button>
                    <button class="control-btn reset" id="reset-btn">🔄 重置</button>
                </div>
                
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
        </div>
    </div>
    
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    
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
        let leftChart = null;
        let rightChart = null;
        
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
                    name: '边缺失影响',
                    description: '测试在不同边缺失率下攻击效果的变化。模拟现实中图数据不完整的情况，观察攻击的鲁棒性。'
                }}
            ],
            defense: [
                {{
                    id: 'differential_privacy',
                    name: '差分隐私',
                    description: '通过添加噪声边和删除部分边来保护隐私。ε参数控制隐私保护强度，ε越小隐私保护越强但效用损失越大。'
                }}
            ]
        }};
        
        // 初始化
        function init() {{
            setupPhaseButtons();
            setupMethodButtons();
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
        
        function updateMethodSelector(phase) {{
            const container = document.getElementById('method-selector');
            container.innerHTML = '';
            
            METHODS[phase].forEach((method, idx) => {{
                const btn = document.createElement('button');
                btn.className = 'method-btn' + (idx === 0 ? ' active' : '');
                btn.textContent = method.name;
                btn.dataset.methodId = method.id;
                btn.addEventListener('click', () => selectMethod(method));
                container.appendChild(btn);
            }});
            
            // 默认选择第一个方法
            if (METHODS[phase].length > 0) {{
                selectMethod(METHODS[phase][0]);
            }}
        }}
        
        function setupMethodButtons() {{
            document.getElementById('play-btn').addEventListener('click', playAnimation);
            document.getElementById('next-btn').addEventListener('click', nextStep);
            document.getElementById('reset-btn').addEventListener('click', resetVisualization);
        }}
        
        function selectMethod(method) {{
            document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            
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
            leftChart = new GraphChart('graph-left', DATA.graphOrig);
            rightChart = new GraphChart('graph-right', DATA.graphAnon);
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
            document.getElementById('graph-left-title').textContent = '原始图';
            document.getElementById('graph-right-title').textContent = '匿名图';
            
            leftChart.updateData(DATA.graphOrig);
            rightChart.updateData(DATA.graphAnon);
            
            // 准备演示步骤
            const candidates = DATA.deanonymization.candidates;
            const groundTruth = DATA.deanonymization.ground_truth;
            const origNodes = Object.keys(candidates);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 演示步骤</h3>';
            
            // 显示前5个节点的匹配过程
            const demoNodes = origNodes.slice(0, 5);
            demoNodes.forEach((node, idx) => {{
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                
                const topCandidate = candidates[node][0];
                const isCorrect = groundTruth[node] === topCandidate.node;
                
                step.innerHTML = `
                    <strong>步骤 ${{idx + 1}}:</strong> 
                    尝试匹配节点 <strong>${{node}}</strong><br>
                    特征: 度=${{DATA.deanonymization.features[node].degree}}, 
                    聚类系数=${{DATA.deanonymization.features[node].clustering.toFixed(3)}}<br>
                    最佳匹配: <strong>${{topCandidate.node}}</strong> 
                    (相似度: ${{(topCandidate.similarity * 100).toFixed(1)}}%)
                    ${{isCorrect ? '✅ 正确' : '❌ 错误'}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            // 更新统计
            updateStats({{
                '节点总数': DATA.graphOrig.nodes.length,
                '匹配对数': demoNodes.length,
                '正确匹配': demoNodes.filter(n => groundTruth[n] === candidates[n][0].node).length,
                '准确率': (demoNodes.filter(n => groundTruth[n] === candidates[n][0].node).length / demoNodes.length * 100).toFixed(1) + '%'
            }});
        }}
        
        function prepareAttributeViz(methodId) {{
            if (!DATA.attribute) {{
                document.getElementById('explanation').innerHTML += 
                    '<p style="color: red;">⚠️ 该数据集不支持属性推断演示</p>';
                return;
            }}
            
            document.getElementById('graph-left-title').textContent = '已知标签节点';
            document.getElementById('graph-right-title').textContent = '标签传播过程';
            
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
            
            leftChart.updateData(graphWithLabels);
            rightChart.updateData(graphWithLabels);
            
            if (methodId === 'neighbor_voting') {{
                prepareNeighborVotingSteps();
            }} else if (methodId === 'label_propagation') {{
                prepareLabelPropagationSteps();
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
                    预测节点 <strong>${{node}}</strong> 的标签<br>
                    邻居投票: ${{Object.entries(pred.votes).map(([l, c]) => `标签${{l}}(${{c}}票)`).join(', ')}}<br>
                    预测结果: <strong>标签 ${{pred.prediction}}</strong>
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
                    更新了 <strong>${{iter.updates.length}}</strong> 个节点的标签<br>
                    ${{iter.updates.slice(0, 3).map(u => 
                        `节点${{u.node}} → 标签${{u.new_label}}`
                    ).join(', ')}}
                    ${{iter.updates.length > 3 ? '...' : ''}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '迭代次数': iterations.length,
                '初始已知': Object.keys(DATA.attribute.known_labels).length,
                '最终标注': Object.keys(DATA.attribute.known_labels).length + 
                    iterations.reduce((sum, iter) => sum + iter.updates.length, 0)
            }});
        }}
        
        function prepareRobustnessViz(methodId) {{
            document.getElementById('graph-left-title').textContent = '完整图';
            document.getElementById('graph-right-title').textContent = '缺失边图 (30%)';
            
            leftChart.updateData(DATA.robustness['0.0']);
            rightChart.updateData(DATA.robustness['0.3']);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 鲁棒性测试</h3>';
            
            Object.keys(DATA.robustness).forEach((ratio, idx) => {{
                const graph = DATA.robustness[ratio];
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>缺失率 ${{(parseFloat(ratio) * 100).toFixed(0)}}%:</strong>
                    保留 ${{graph.links.length}} 条边
                    (原始: ${{DATA.robustness['0.0'].links.length}} 条)
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '原始节点数': DATA.robustness['0.0'].nodes.length,
                '原始边数': DATA.robustness['0.0'].links.length,
                '测试场景': Object.keys(DATA.robustness).length
            }});
        }}
        
        function prepareDefenseViz(methodId) {{
            document.getElementById('graph-left-title').textContent = '原始图';
            document.getElementById('graph-right-title').textContent = '防御后 (ε=1.0)';
            
            leftChart.updateData(DATA.graphOrig);
            rightChart.updateData(DATA.defense['1.0']);
            
            const stepsContainer = document.getElementById('steps-container');
            stepsContainer.innerHTML = '<h3>🎬 差分隐私防御</h3>';
            
            Object.keys(DATA.defense).forEach((epsilon, idx) => {{
                const graph = DATA.defense[epsilon];
                const addedEdges = graph.links.length - DATA.graphOrig.links.length;
                
                const step = document.createElement('div');
                step.className = 'step';
                step.id = `step-${{idx}}`;
                step.innerHTML = `
                    <strong>ε = ${{epsilon}}:</strong>
                    添加 ${{addedEdges}} 条噪声边
                    (总边数: ${{graph.links.length}})
                    <br>隐私强度: ${{parseFloat(epsilon) < 1 ? '强' : parseFloat(epsilon) < 2 ? '中' : '弱'}}
                `;
                stepsContainer.appendChild(step);
            }});
            
            updateStats({{
                '原始边数': DATA.graphOrig.links.length,
                '防御方案': Object.keys(DATA.defense).length,
                'ε范围': `${{Math.min(...Object.keys(DATA.defense).map(parseFloat))}} - ${{Math.max(...Object.keys(DATA.defense).map(parseFloat))}}`
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
                nextStep();
                if (currentStep >= document.querySelectorAll('.step').length) {{
                    stopAnimation();
                }}
            }}, 2000);
        }}
        
        function stopAnimation() {{
            isPlaying = false;
            document.getElementById('play-btn').innerHTML = '▶️ 开始演示';
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
                
                // 高亮相关节点
                highlightStep(currentStep);
                
                currentStep++;
            }}
        }}
        
        function highlightStep(stepIdx) {{
            // 根据当前阶段和方法高亮不同的节点
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
                
                leftChart.highlightNodes([origNode], topCandidates);
                rightChart.highlightNodes(topCandidates, [origNode]);
            }}
        }}
        
        function highlightAttributeStep(stepIdx) {{
            if (!DATA.attribute) return;
            
            if (currentMethod.id === 'neighbor_voting') {{
                const nodes = Object.keys(DATA.attribute.neighbor_predictions).slice(0, 5);
                if (stepIdx < nodes.length) {{
                    const node = nodes[stepIdx];
                    const neighbors = DATA.attribute.neighbor_predictions[node].neighbors;
                    leftChart.highlightNodes([node], neighbors);
                    rightChart.highlightNodes([node], neighbors);
                }}
            }} else if (currentMethod.id === 'label_propagation') {{
                const iterations = DATA.attribute.label_propagation;
                if (stepIdx < iterations.length) {{
                    const updates = iterations[stepIdx].updates;
                    const updatedNodes = updates.map(u => u.node);
                    leftChart.highlightNodes(updatedNodes, []);
                    rightChart.highlightNodes(updatedNodes, []);
                }}
            }}
        }}
        
        function highlightRobustnessStep(stepIdx) {{
            const ratios = Object.keys(DATA.robustness);
            if (stepIdx < ratios.length) {{
                const ratio = ratios[stepIdx];
                rightChart.updateData(DATA.robustness[ratio]);
                document.getElementById('graph-right-title').textContent = 
                    `缺失边图 (${{(parseFloat(ratio) * 100).toFixed(0)}}%)`;
            }}
        }}
        
        function highlightDefenseStep(stepIdx) {{
            const epsilons = Object.keys(DATA.defense);
            if (stepIdx < epsilons.length) {{
                const epsilon = epsilons[stepIdx];
                rightChart.updateData(DATA.defense[epsilon]);
                document.getElementById('graph-right-title').textContent = 
                    `防御后 (ε=${{epsilon}})`;
            }}
        }}
        
        function resetVisualization() {{
            stopAnimation();
            currentStep = 0;
            document.querySelectorAll('.step').forEach(s => s.classList.remove('current'));
            leftChart.resetHighlights();
            rightChart.resetHighlights();
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
                
                // 添加缩放
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
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
                
                // 创建力导向布局
                this.simulation = d3.forceSimulation(this.data.nodes)
                    .force('link', d3.forceLink(this.data.links)
                        .id(d => d.index)
                        .distance(50))
                    .force('charge', d3.forceManyBody().strength(-200))
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .force('collision', d3.forceCollide().radius(20));
                
                // 绘制边
                this.links = this.g.append('g')
                    .selectAll('line')
                    .data(this.data.links)
                    .join('line')
                    .attr('class', 'link')
                    .attr('stroke-width', 1.5);
                
                // 绘制节点
                this.nodes = this.g.append('g')
                    .selectAll('circle')
                    .data(this.data.nodes)
                    .join('circle')
                    .attr('class', 'node')
                    .attr('r', d => 5 + Math.sqrt(d.degree || 1) * 2)
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
                
                // 添加标签
                this.labels = this.g.append('g')
                    .selectAll('text')
                    .data(this.data.nodes)
                    .join('text')
                    .text(d => d.id)
                    .attr('font-size', 10)
                    .attr('dx', 12)
                    .attr('dy', 4)
                    .style('pointer-events', 'none');
                
                this.simulation.on('tick', () => {{
                    this.links
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    this.nodes
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                    
                    this.labels
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                }});
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
            
            highlightNodes(primaryNodes, secondaryNodes = []) {{
                this.resetHighlights();
                
                this.nodes
                    .classed('highlighted', d => primaryNodes.includes(d.id))
                    .classed('candidate', d => secondaryNodes.includes(d.id));
            }}
            
            resetHighlights() {{
                if (this.nodes) {{
                    this.nodes
                        .classed('highlighted', false)
                        .classed('candidate', false)
                        .classed('matched', false);
                }}
                if (this.links) {{
                    this.links.classed('highlighted', false);
                }}
            }}
            
            showTooltip(event, d) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
                tooltip.innerHTML = `
                    <strong>节点 ${{d.id}}</strong><br>
                    度数: ${{d.degree || 0}}<br>
                    ${{d.label !== undefined ? `标签: ${{d.label}}` : ''}}
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
    
    parser = argparse.ArgumentParser(description="生成攻击原理交互式演示")
    parser.add_argument('--ego_id', type=str, default='698',
                       help='Ego网络ID (建议使用小规模网络如698)')
    parser.add_argument('--output', type=str, default='results/attack_principles_demo.html',
                       help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    print("="*70)
    print("攻击原理可视化演示工具")
    print("="*70)
    
    visualizer = AttackPrincipleVisualizer(ego_id=args.ego_id)
    output_file = visualizer.generate_html(output_file=args.output)
    
    print("\n" + "="*70)
    print("✅ 生成完成！")
    print(f"📂 文件位置: {output_file}")
    print("\n使用说明:")
    print("1. 在浏览器中打开HTML文件")
    print("2. 选择攻击阶段和具体方法")
    print("3. 点击'开始演示'观看动画")
    print("4. 可以拖动节点、缩放图形")
    print("5. 鼠标悬停在节点上查看详细信息")
    print("="*70)


if __name__ == "__main__":
    main()

