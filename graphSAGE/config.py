import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


class Config:
    # 路径配置
    ORIG_GRAPH_PATH = DATA_DIR / "processed/graph.gpickle"
    ANON_GRAPH_PATH = DATA_DIR / "anonymized/anonymous_graph.gpickle"
    GT_PATH = DATA_DIR / "anonymized/ground_truth.pkl"
    MODEL_SAVE_PATH = Path(__file__).parent / "saved_models/sage_align.pth"

    FEATURE_KEYS = [
        'degree_centrality',
        'betweenness_centrality',
        'closeness_centrality',
        'pagerank',
        'clustering',
        'in_degree',
        'out_degree'
    ]

    SEED_RATIO = 0.05
    HIDDEN_DIM = 128
    OUTPUT_DIM = 64
    LEARNING_RATE = 0.005
    EPOCHS = 200
    DROPOUT = 0.5