import torch
import json
import os
from pathlib import Path
from config import Config
from dataloader import GraphLoader
from network import GraphSAGE


def evaluate():
    cfg = Config()
    loader = GraphLoader()

    data_orig, data_anon, train_pairs, test_pairs = loader.load()

    input_dim = data_orig.x.shape[1]
    model = GraphSAGE(input_dim, cfg.HIDDEN_DIM, cfg.OUTPUT_DIM)

    # 检查模型是否存在
    if not cfg.MODEL_SAVE_PATH.exists():
        print(f"❌ 未找到模型文件: {cfg.MODEL_SAVE_PATH}")
        print("   请先运行 train.py 进行训练")
        return

    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
    model.eval()

    print("生成最终嵌入...")
    with torch.no_grad():
        emb_orig, emb_anon = model(data_orig, data_anon)

    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    hits_20 = 0
    total = len(test_pairs)

    orig_indices = test_pairs[:, 0]
    true_anon_indices = test_pairs[:, 1]

    # 批量计算相似度矩阵
    target_embs = emb_orig[orig_indices]
    all_anon_embs = emb_anon
    sim_matrix = torch.mm(target_embs, all_anon_embs.t())

    print("正在计算排名 (Top-20)...")
    for i in range(total):
        true_idx = true_anon_indices[i].item()
        scores = sim_matrix[i]

        # 获取前20名
        _, top_k_indices = torch.topk(scores, k=20)
        top_k_list = top_k_indices.tolist()

        if true_idx == top_k_list[0]:
            hits_1 += 1
        if true_idx in top_k_list[:5]:
            hits_5 += 1
        if true_idx in top_k_list[:10]:
            hits_10 += 1
        if true_idx in top_k_list[:20]:
            hits_20 += 1

    # --- 计算指标 ---
    accuracy = hits_1 / total

    print("=" * 30)
    print(f"测试集大小: {total}")
    print(f"Hit@1  (Acc): {accuracy:.2%}")
    print(f"Hit@5       : {hits_5 / total:.2%}")
    print(f"Hit@10      : {hits_10 / total:.2%}")
    print(f"Hit@20      : {hits_20 / total:.2%}")
    print("=" * 30)

    # --- 保存结果 ---
    save_results_to_json(accuracy, hits_1, hits_5, hits_10, hits_20, total)


def save_results_to_json(acc, h1, h5, h10, h20, total):
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"

    if not results_dir.exists():
        print(f"📂 创建目录: {results_dir}")
        results_dir.mkdir(parents=True, exist_ok=True)

    file_path = results_dir / "attack_results.json"

    current_result = {
        "accuracy": acc,
        "precision": acc,
        "recall": acc,
        "f1": acc,
        "top_k": {
            "1": h1 / total,
            "5": h5 / total,
            "10": h10 / total,
            "20": h20 / total
        }
    }

    final_data = {}
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ JSON 文件损坏或为空，将创建新文件。")
            final_data = {}

    final_data["GraphSAGE"] = current_result

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 结果已成功写入: {file_path}")


if __name__ == "__main__":
    evaluate()