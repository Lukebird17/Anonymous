import pickle
import torch
import random
from pathlib import Path
from torch_geometric.data import Data

from config import Config
from utils import compute_features


class GraphLoader:
    def __init__(self):
        self.cfg = Config()


    def load(self):
        with open(self.cfg.ORIG_GRAPH_PATH, 'rb') as f:
            G_orig = pickle.load(f)
        with open(self.cfg.ANON_GRAPH_PATH, 'rb') as f:
            G_anon = pickle.load(f)

        with open(self.cfg.GT_PATH, 'rb') as f:
            ground_truth = pickle.load(f)['node_mapping']

        print(f"[Loader] 数据加载成功。Ground Truth 大小: {len(ground_truth)} 对")

        print(f"[Loader] 正在计算图特征...")
        x1, edge1, map1 = compute_features(G_orig, self.cfg.FEATURE_KEYS)
        x2, edge2, map2 = compute_features(G_anon, self.cfg.FEATURE_KEYS)

        data_orig = Data(x=x1, edge_index=edge1)
        data_anon = Data(x=x2, edge_index=edge2)

        all_pairs = []
        match_count = 0
        miss_count = 0

        for orig_id, anon_id in ground_truth.items():

            if orig_id in map1 and anon_id in map2:
                idx1 = map1[orig_id]
                idx2 = map2[anon_id]
                all_pairs.append((idx1, idx2))
                match_count += 1
            else:
                miss_count += 1

        print(f"   [匹配结果] 成功: {match_count} 对 | 丢失: {miss_count} 对 (孤立点或已被删除的节点)")

        if len(all_pairs) == 0:
            raise RuntimeError("❌ 致命错误: 匹配数量为 0，请检查数据完整性！")

        random.seed(42)
        random.shuffle(all_pairs)

        split_idx = int(len(all_pairs) * self.cfg.SEED_RATIO)
        if split_idx == 0 and len(all_pairs) > 0:
            split_idx = 1
            print("⚠️ 提示: 种子比例较低，强制划分为 1 个训练样本")

        train_pairs = torch.tensor(all_pairs[:split_idx], dtype=torch.long)
        test_pairs = torch.tensor(all_pairs[split_idx:], dtype=torch.long)

        print(f"[Loader] 划分完成: 训练集 {len(train_pairs)} (种子), 测试集 {len(test_pairs)}")

        return data_orig, data_anon, train_pairs, test_pairs