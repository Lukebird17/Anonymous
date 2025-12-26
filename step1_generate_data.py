#!/usr/bin/env python3
"""
生成示例社交网络数据
无需爬虫，直接生成用于测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx
import json

def generate_example_network(n_nodes=1000, avg_degree=5):
    """生成示例社交网络（Barabasi-Albert模型）"""
    print(f"生成示例社交网络...")
    print(f"  节点数: {n_nodes}")
    print(f"  平均度: {avg_degree}")
    
    # 生成无标度网络（类似真实社交网络）
    G = nx.barabasi_albert_graph(n_nodes, avg_degree)
    
    print(f"\n✅ 生成完成:")
    print(f"   节点数: {G.number_of_nodes()}")
    print(f"   边数: {G.number_of_edges()}")
    print(f"   平均度: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    
    # 转换为项目所需的JSON格式
    data = {
        'users': {
            str(node): {
                'uid': str(node),
                'screen_name': f'User_{node}',
                'followers_count': G.degree(node),
                'follow_count': G.degree(node),
            }
            for node in G.nodes()
        },
        'edges': [(str(u), str(v)) for u, v in G.edges()],
        'metadata': {
            'source': 'Generated Barabasi-Albert Graph',
            'model': 'ba_graph',
            'total_users': G.number_of_nodes(),
            'total_edges': G.number_of_edges()
        }
    }
    
    return data


def main():
    # 创建数据目录
    data_dir = Path(__file__).parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成数据
    data = generate_example_network(n_nodes=1000, avg_degree=5)
    
    # 保存
    output_path = data_dir / 'example_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 数据已保存到: {output_path}")
    print(f"\n📌 下一步:")
    print(f"   python step2_build_graph.py")


if __name__ == "__main__":
    main()


