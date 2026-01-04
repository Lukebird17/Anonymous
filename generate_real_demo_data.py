#!/usr/bin/env python3
"""
从真实实验结果生成可视化演示数据
"""
import json
import networkx as nx
import numpy as np
from pathlib import Path
import argparse

def generate_synthetic_graph(results):
    """从实验结果生成模拟图"""
    print("  🔄 生成模拟图...")
    stats = results.get('graph_stats', {})
    n_nodes = stats.get('nodes', 50)
    n_edges = stats.get('edges', 200)
    avg_degree = stats.get('avg_degree', 8)
    
    # 限制节点数
    n_nodes = min(n_nodes, 100)
    
    # 使用BA模型生成无标度网络
    m = max(1, int(avg_degree / 2))
    G = nx.barabasi_albert_graph(n_nodes, m, seed=42)
    
    print(f"  ✅ 生成了 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    
    # 随机分配属性
    for node in G.nodes():
        G.nodes[node]['attribute'] = np.random.choice(['A', 'B', 'C'])
    
    return G

def load_graph(dataset, ego_id=None, results=None):
    """加载图数据"""
    if dataset == 'facebook_ego':
        if ego_id is None:
            ego_id = '0'
        
        # 尝试多个可能的数据路径
        possible_paths = [
            Path('data/datasets/facebook'),
            Path('data/facebook'),
            Path('data'),
            Path('../data/datasets/facebook'),
            Path('../data/facebook'),
            Path('../../data/datasets/facebook'),
        ]
        
        edge_file = None
        feat_file = None
        
        for data_dir in possible_paths:
            edge_candidate = data_dir / f'{ego_id}.edges'
            feat_candidate = data_dir / f'{ego_id}.feat'
            if edge_candidate.exists():
                edge_file = edge_candidate
                feat_file = feat_candidate
                print(f"  📁 找到数据文件: {edge_file}")
                break
        
        if edge_file is None:
            print(f"  ⚠️  警告: 找不到边文件，将使用实验结果中的统计信息生成模拟图")
            # 从实验结果生成模拟图
            if results:
                return generate_synthetic_graph(results)
            else:
                raise FileNotFoundError(f"找不到 {ego_id}.edges 文件，也没有提供results参数")
        
        G = nx.Graph()
        
        # 读取边
        edge_count = 0
        with open(edge_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        G.add_edge(int(parts[0]), int(parts[1]))
                        edge_count += 1
                    except ValueError:
                        continue
        
        print(f"  ✅ 读取了 {edge_count} 条边")
        
        # 读取特征（作为属性）
        if feat_file and feat_file.exists():
            attr_count = 0
            with open(feat_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            node_id = int(parts[0])
                            # 简化：使用特征的前几个维度来确定属性类别
                            features = [int(x) for x in parts[1:]]
                            attr_sum = sum(features[:3]) if len(features) >= 3 else 0
                            if attr_sum < 1:
                                attr = 'A'
                            elif attr_sum < 2:
                                attr = 'B'
                            else:
                                attr = 'C'
                            if node_id in G.nodes():
                                G.nodes[node_id]['attribute'] = attr
                                attr_count += 1
                        except ValueError:
                            continue
            print(f"  ✅ 读取了 {attr_count} 个节点的属性")
        else:
            print(f"  ⚠️  特征文件不存在，将随机分配属性")
            # 随机分配属性
            for node in G.nodes():
                G.nodes[node]['attribute'] = np.random.choice(['A', 'B', 'C'])
    
    elif dataset == 'cora':
        # 加载Cora数据集
        try:
            from torch_geometric.datasets import Planetoid
            import torch_geometric.transforms as T
            
            print(f"  📦 加载Cora数据集...")
            dataset_obj = Planetoid(root='data', name='Cora', transform=T.NormalizeFeatures())
            data = dataset_obj[0]
            
            G = nx.Graph()
            edge_index = data.edge_index.numpy()
            edges = list(zip(edge_index[0], edge_index[1]))
            G.add_edges_from(edges)
            print(f"  ✅ 加载了 {len(edges)} 条边")
            
            # 添加属性标签
            labels = data.y.numpy()
            attr_count = 0
            for node in G.nodes():
                if node < len(labels):
                    label = int(labels[node])
                    if label == 0:
                        attr = 'A'
                    elif label in [1, 2]:
                        attr = 'B'
                    else:
                        attr = 'C'
                    G.nodes[node]['attribute'] = attr
                    attr_count += 1
            print(f"  ✅ 设置了 {attr_count} 个节点的属性")
            
        except ImportError:
            print(f"  ⚠️  警告: torch_geometric未安装，将生成模拟图")
            if results:
                return generate_synthetic_graph(results)
            else:
                raise ImportError("需要安装 torch_geometric 或提供 results 参数")
        except Exception as e:
            print(f"  ⚠️  警告: 加载Cora失败 ({e})，将生成模拟图")
            if results:
                return generate_synthetic_graph(results)
            else:
                raise
    
    else:
        print(f"  ⚠️  未知数据集: {dataset}，将生成模拟图")
        if results:
            return generate_synthetic_graph(results)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    return G

def compute_layout(G, max_nodes=50):
    """计算图布局（限制节点数以提高性能）"""
    if len(G.nodes()) == 0:
        raise ValueError("图为空，无法计算布局")
    
    # 如果节点太多，采样一个子图
    if len(G.nodes()) > max_nodes:
        # 选择度数最高的节点
        nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        selected_nodes = [n for n, d in nodes_by_degree[:max_nodes]]
        G_sub = G.subgraph(selected_nodes).copy()
    else:
        G_sub = G
    
    if len(G_sub.nodes()) == 0:
        raise ValueError("子图为空，无法计算布局")
    
    # 使用spring layout
    pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
    
    if len(pos) == 0:
        raise ValueError("布局计算失败，没有节点位置")
    
    # 归一化到[0, 600]范围
    pos_array = np.array(list(pos.values()))
    min_pos = pos_array.min(axis=0)
    max_pos = pos_array.max(axis=0)
    
    pos_normalized = {}
    for node, (x, y) in pos.items():
        x_norm = (x - min_pos[0]) / (max_pos[0] - min_pos[0]) * 500 + 50
        y_norm = (y - min_pos[1]) / (max_pos[1] - min_pos[1]) * 500 + 50
        pos_normalized[node] = (x_norm, y_norm)
    
    return pos_normalized, G_sub

def generate_greedy_steps(G, results, max_steps=10):
    """生成贪心匹配的演示步骤"""
    steps = []
    nodes = list(G.nodes())[:max_steps]
    
    # 找到对应的结果
    greedy_result = None
    for r in results.get('deanonymization', []):
        if 'Greedy' in r['method']:
            greedy_result = r
            break
    
    if greedy_result is None:
        return []
    
    accuracy = greedy_result.get('accuracy', 0.5)
    
    for i, node in enumerate(nodes):
        # 模拟匹配过程
        success = np.random.random() < accuracy
        steps.append({
            'orig_node': int(node),
            'anon_node': int(node),  # 简化：假设相同
            'success': success,
            'similarity': float(np.random.random() * 0.5 + 0.5) if success else float(np.random.random() * 0.5),
            'description': f'匹配节点 {node}: {"成功" if success else "失败"}'
        })
    
    return steps

def generate_deepwalk_walks(G, n_walks=3, walk_length=5):
    """生成随机游走演示"""
    walks = []
    nodes = list(G.nodes())
    
    for _ in range(n_walks):
        if not nodes:
            break
        start_node = np.random.choice(nodes)
        walk = [int(start_node)]
        current = start_node
        
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = np.random.choice(neighbors)
            walk.append(int(current))
        
        walks.append(walk)
    
    return walks

def generate_attribute_inference_steps(G, results, max_steps=8):
    """生成属性推断演示步骤"""
    steps = []
    
    # 找到邻居投票的结果
    attr_result = None
    for r in results.get('attribute_inference', []):
        if 'Voting' in r['method']:
            attr_result = r
            break
    
    if attr_result is None:
        return []
    
    accuracy = attr_result.get('accuracy', 0.5)
    
    # 选择一些没有属性的节点
    nodes_with_attr = [n for n in G.nodes() if 'attribute' in G.nodes[n]]
    nodes_without_attr = [n for n in G.nodes() if 'attribute' not in G.nodes[n]]
    
    if not nodes_without_attr:
        # 如果都有属性，随机隐藏一些
        nodes_without_attr = np.random.choice(nodes_with_attr, min(max_steps, len(nodes_with_attr)), replace=False)
    
    for node in list(nodes_without_attr)[:max_steps]:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        
        # 统计邻居属性
        neighbor_attrs = []
        for n in neighbors:
            if 'attribute' in G.nodes[n]:
                neighbor_attrs.append(G.nodes[n]['attribute'])
        
        if not neighbor_attrs:
            continue
        
        # 投票
        from collections import Counter
        votes = Counter(neighbor_attrs)
        predicted = votes.most_common(1)[0][0]
        
        steps.append({
            'node': int(node),
            'neighbors': [int(n) for n in neighbors[:5]],  # 最多显示5个邻居
            'votes': dict(votes),
            'predicted': predicted,
            'correct': np.random.random() < accuracy
        })
    
    return steps

def generate_defense_data(G, results):
    """生成防御演示数据"""
    defense_result = None
    for r in results.get('defense', []):
        if r.get('epsilon') == 0.1:  # 使用epsilon=0.1的结果
            defense_result = r
            break
    
    if defense_result is None:
        return {'edges_to_remove': [], 'edges_to_add': []}
    
    edges = list(G.edges())
    
    # 根据structural_loss选择要删除和添加的边
    n_remove = min(10, len(edges) // 10)
    n_add = min(15, len(edges) // 10)
    
    edges_to_remove = list(np.random.choice(len(edges), n_remove, replace=False))
    
    # 生成要添加的边（随机节点对）
    nodes = list(G.nodes())
    edges_to_add = []
    for _ in range(n_add):
        n1, n2 = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(n1, n2):
            edges_to_add.append({'source': int(n1), 'target': int(n2)})
    
    return {
        'edges_to_remove': edges_to_remove,
        'edges_to_add': edges_to_add
    }

def graph_to_json(G, pos):
    """将图转换为JSON格式"""
    nodes = []
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        nodes.append({
            'id': int(node),
            'index': int(node),
            'x': float(x),
            'y': float(y),
            'degree': G.degree(node),
            'attribute': G.nodes[node].get('attribute', 'Unknown'),
            'known': bool(np.random.random() > 0.3)  # 30%未知
        })
    
    links = []
    for u, v in G.edges():
        if u in pos and v in pos:
            links.append({
                'source': int(u),
                'target': int(v)
            })
    
    return {'nodes': nodes, 'links': links}

def generate_graph_kernel_data(G):
    """生成图核方法的演示数据"""
    nodes = list(G.nodes())
    if not nodes:
        return {}
    
    center_node = np.random.choice(nodes)
    neighbors = list(G.neighbors(center_node))
    
    return {
        'center_node': int(center_node),
        'hops': [
            {'nodes': [int(n) for n in neighbors[:5]]}
        ]
    }

def main():
    parser = argparse.ArgumentParser(description='从实验结果生成可视化演示数据')
    parser.add_argument('--result_file', type=str, required=True,
                        help='实验结果JSON文件路径')
    parser.add_argument('--output', type=str, default='results/real_demo_data_final.json',
                        help='输出JSON文件路径')
    parser.add_argument('--max_nodes', type=int, default=50,
                        help='最大显示节点数（默认50）')
    
    args = parser.parse_args()
    
    print(f"📖 读取实验结果: {args.result_file}")
    with open(args.result_file) as f:
        results = json.load(f)
    
    dataset = results['dataset']
    ego_id = results.get('ego_id')
    
    print(f"📊 数据集: {dataset}, Ego ID: {ego_id}")
    print(f"📈 图统计: {results['graph_stats']}")
    
    # 加载图
    print("🔄 加载图数据...")
    G = load_graph(dataset, ego_id, results)
    print(f"✅ 图加载完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    if G.number_of_nodes() == 0:
        print("❌ 错误: 图为空，无法继续")
        return
    
    # 计算布局
    print("🎨 计算图布局...")
    pos, G_sub = compute_layout(G, max_nodes=args.max_nodes)
    print(f"✅ 使用 {len(G_sub.nodes())} 个节点进行可视化")
    
    # 转换为JSON格式
    print("🔄 生成图数据...")
    graph_data = graph_to_json(G_sub, pos)
    
    # 生成动画数据
    print("🎬 生成动画数据...")
    
    print("  - 贪心匹配...")
    greedy_steps = generate_greedy_steps(G_sub, results)
    
    print("  - 匈牙利算法...")
    hungarian_steps = greedy_steps[:5]  # 使用前5步作为示例
    
    print("  - 图核方法...")
    graph_kernel_data = generate_graph_kernel_data(G_sub)
    
    print("  - DeepWalk...")
    deepwalk_walks = generate_deepwalk_walks(G_sub)
    
    print("  - 属性推断...")
    attribute_steps = generate_attribute_inference_steps(G_sub, results)
    
    print("  - 防御方法...")
    defense_data = generate_defense_data(G_sub, results)
    
    # 组装最终数据
    demo_data = {
        'meta': {
            'dataset': dataset,
            'ego_id': ego_id,
            'nodes': len(graph_data['nodes']),
            'edges': len(graph_data['links']),
            'timestamp': results['timestamp']
        },
        'graph': graph_data,
        'results': {
            'deanonymization': results['deanonymization'],
            'attribute_inference': results['attribute_inference'],
            'defense': results['defense']
        },
        'animations': {
            'greedy': greedy_steps,
            'hungarian': hungarian_steps,
            'graph_kernel': graph_kernel_data,
            'deepwalk': {
                'walks': deepwalk_walks
            },
            'attribute_inference': attribute_steps,
            'defense': defense_data
        }
    }
    
    # 保存
    print(f"💾 保存到: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    demo_data = convert_numpy(demo_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print("✅ 完成！")
    print(f"\n📊 生成的数据统计:")
    print(f"  - 节点数: {len(graph_data['nodes'])}")
    print(f"  - 边数: {len(graph_data['links'])}")
    print(f"  - 贪心步骤: {len(greedy_steps)}")
    print(f"  - 随机游走: {len(deepwalk_walks)}")
    print(f"  - 属性推断步骤: {len(attribute_steps)}")
    print(f"  - 去匿名化方法: {len(results['deanonymization'])}")
    print(f"  - 属性推断方法: {len(results['attribute_inference'])}")
    print(f"  - 防御方法: {len(results['defense'])}")
    
    print(f"\n🚀 使用方法:")
    print(f"  1. 将生成的数据文件复制到网页同目录")
    print(f"  2. 修改 animated_attack_demo.html 中的数据路径为: {args.output}")
    print(f"  3. 运行: ./run_animated_demo.sh")

if __name__ == '__main__':
    main()

