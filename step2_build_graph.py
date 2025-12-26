#!/usr/bin/env python3
"""
构建图并计算特征
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.graph_builder import GraphBuilder
import pickle

def main():
    print("="*60)
    print("步骤2: 构建图并计算特征")
    print("="*60)
    
    # 初始化构建器
    builder = GraphBuilder()
    
    # 加载数据并构建图
    data_path = Path(__file__).parent / 'data' / 'raw' / 'example_data.json'
    
    if not data_path.exists():
        print(f"\n❌ 错误: 找不到数据文件 {data_path}")
        print(f"   请先运行: python step1_generate_data.py")
        return
    
    print(f"\n📂 加载数据: {data_path}")
    G = builder.build_from_github(data_path, use_starred_repos=False)
    
    print(f"\n🔢 计算节点特征...")
    G = builder.compute_node_features(G)
    
    print(f"\n🔍 提取最大连通分量...")
    G = builder.extract_largest_component(G)
    
    # 打印统计信息
    builder.print_graph_stats(G)
    
    # 保存
    output_path = Path(__file__).parent / 'data' / 'processed' / 'graph.gpickle'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"💾 图已保存到: {output_path}")
    print(f"\n📌 下一步:")
    print(f"   python step3_anonymize.py")


if __name__ == "__main__":
    main()


