#!/usr/bin/env python3
"""
匿名化处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.anonymizer import GraphAnonymizer
import pickle

def main():
    print("="*60)
    print("步骤3: 匿名化处理")
    print("="*60)
    
    # 加载原始图
    graph_path = Path(__file__).parent / 'data' / 'processed' / 'graph.gpickle'
    
    if not graph_path.exists():
        print(f"\n❌ 错误: 找不到图文件 {graph_path}")
        print(f"   请先运行: python step2_build_graph.py")
        return
    
    print(f"\n📂 加载图: {graph_path}")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"   节点数: {G.number_of_nodes()}")
    print(f"   边数: {G.number_of_edges()}")
    
    # 匿名化
    print(f"\n🔒 执行匿名化...")
    anonymizer = GraphAnonymizer(
        edge_retention_ratio=0.7,    # 保留70%的边
        add_noise_edges=True,         # 添加噪声边
        noise_ratio=0.05              # 5%噪声
    )
    
    G_anon, node_mapping = anonymizer.anonymize(G)
    
    print(f"\n✅ 匿名化完成:")
    print(f"   原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"   匿名图: {G_anon.number_of_nodes()} 节点, {G_anon.number_of_edges()} 边")
    print(f"   边保留率: {G_anon.number_of_edges()/G.number_of_edges():.1%}")
    
    # 创建ground truth
    ground_truth = anonymizer.create_ground_truth(G, G_anon, node_mapping)
    
    # 保存
    output_dir = Path(__file__).parent / 'data' / 'anonymized'
    anonymizer.save_anonymized_data(G_anon, ground_truth, output_dir)
    
    print(f"\n💾 匿名化数据已保存到: {output_dir}")
    print(f"\n📌 下一步:")
    print(f"   python step4_attack.py")


if __name__ == "__main__":
    main()


