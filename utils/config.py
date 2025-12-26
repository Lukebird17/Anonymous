"""
配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANONYMIZED_DATA_DIR = DATA_DIR / "anonymized"

# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    "deepwalk": {
        "dimensions": 128,
        "walk_length": 80,
        "num_walks": 10,
        "window_size": 10,
        "workers": 4,
        "epochs": 5
    },
    "graphsage": {
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "learning_rate": 0.01,
        "epochs": 100,
        "batch_size": 256
    }
}

# 爬虫配置
CRAWLER_CONFIG = {
    "weibo": {
        "cookies": None,  # 需要用户自行配置
        "max_depth": 3,
        "max_users": 5000,
        "delay": 1.0,  # 请求间隔（秒）
    },
    "github": {
        "token": None,  # 需要用户自行配置
        "max_users": 5000,
        "api_delay": 0.5,
    },
    "bilibili": {
        "cookies": None,
        "max_users": 5000,
        "delay": 1.0,
    }
}

# 匿名化配置
ANONYMIZATION_CONFIG = {
    "edge_retention_ratio": 0.7,  # 保留边的比例
    "add_noise_edges": False,     # 是否添加噪声边
    "noise_ratio": 0.05,          # 噪声边比例
    "k_anonymity": None,          # k-匿名度（None表示不使用）
}

# 攻击配置
ATTACK_CONFIG = {
    "seed_ratio": 0.05,           # 种子节点比例
    "top_k": [1, 5, 10, 20],      # Top-K评估
    "matching_method": "cosine",  # 相似度计算方法
    "alignment_method": "procrustes",  # 图对齐方法
}

# 可视化配置
VIZ_CONFIG = {
    "node_size": 10,
    "edge_width": 0.5,
    "figure_size": (12, 10),
    "layout": "spring",  # spring, kamada_kawai, circular
}

# 确保目录存在
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  ANONYMIZED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


