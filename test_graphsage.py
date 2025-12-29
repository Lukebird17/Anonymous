"""
GraphSAGE快速测试脚本
用于验证GraphSAGE实现是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from attack.graphsage_attribute_inference import GraphSAGEAttributeInferenceAttack
from data.dataset_loader import DatasetLoader
import torch


def test_graphsage_cora():
    """在Cora数据集上测试GraphSAGE"""
    print("="*70)
    print("在Cora数据集上测试GraphSAGE")
    print("="*70)
    
    # 检查PyTorch
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 加载数据
    print("\n加载Cora数据集...")
    loader = DatasetLoader()
    G, attributes = loader.load_cora()
    
    if G is None:
        print("❌ 加载失败")
        return
    
    print(f"✅ 加载成功")
    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")
    
    # 创建攻击器
    print("\n创建GraphSAGE攻击器...")
    attacker = GraphSAGEAttributeInferenceAttack(G, attributes)
    
    # 运行攻击
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = attacker.run_attack(
        train_ratio=0.3,
        epochs=50,
        batch_size=64,
        hidden_dim=64,
        embed_dim=32,
        learning_rate=0.01,
        device=device
    )
    
    # 打印结果
    print("\n"+"="*70)
    print("最终结果")
    print("="*70)
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    print(f"F1 (micro): {results['f1_micro']:.4f}")
    print(f"训练集: {results['train_nodes']}个节点")
    print(f"测试集: {results['test_nodes']}个节点")
    print(f"类别数: {results['num_classes']}")
    
    # 与标签传播对比
    print("\n"+"="*70)
    print("预期结果对比")
    print("="*70)
    print("标签传播: ~82-83%准确率")
    print("GraphSAGE: ~80-90%准确率（取决于参数和随机性）")
    
    if results['accuracy'] > 0.75:
        print("\n✅ GraphSAGE效果很好！")
    elif results['accuracy'] > 0.60:
        print("\n⚠️ GraphSAGE效果中等，可能需要调参")
    else:
        print("\n❌ GraphSAGE效果较差，请检查实现")


def test_graphsage_facebook():
    """在Facebook Ego网络上测试GraphSAGE"""
    print("="*70)
    print("在Facebook Ego-0网络上测试GraphSAGE")
    print("="*70)
    
    # 加载数据
    print("\n加载Facebook Ego-0数据集...")
    loader = DatasetLoader()
    G, attributes = loader.load_facebook(ego_network='0')
    
    if G is None:
        print("❌ 加载失败")
        return
    
    print(f"✅ 加载成功")
    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")
    
    # 创建攻击器
    print("\n创建GraphSAGE攻击器...")
    attacker = GraphSAGEAttributeInferenceAttack(G, attributes)
    
    # 运行攻击
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = attacker.run_attack(
        train_ratio=0.3,
        epochs=30,  # Facebook网络较小，训练轮数少一些
        batch_size=32,
        hidden_dim=64,
        embed_dim=32,
        learning_rate=0.01,
        device=device
    )
    
    # 打印结果
    print("\n"+"="*70)
    print("最终结果")
    print("="*70)
    print(f"准确率: {results['accuracy']:.2%}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    print(f"训练集: {results['train_nodes']}个节点")
    print(f"测试集: {results['test_nodes']}个节点")
    print(f"类别数: {results['num_classes']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试GraphSAGE实现')
    parser.add_argument('--dataset', type=str, default='cora',
                       choices=['cora', 'facebook'],
                       help='数据集选择')
    args = parser.parse_args()
    
    if args.dataset == 'cora':
        test_graphsage_cora()
    elif args.dataset == 'facebook':
        test_graphsage_facebook()

