import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from pathlib import Path

from config import Config
from dataloader import GraphLoader
from network import GraphSAGE


class Trainer:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")

        self._prepare_data()
        self._build_model()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE
        )

    def _prepare_data(self):
        """内部方法: 加载并处理数据到指定设备"""
        print("\n[Data] 正在加载数据...")
        loader = GraphLoader()
        data_orig, data_anon, train_pairs, test_pairs = loader.load()

        self.data_orig = data_orig.to(self.device)
        self.data_anon = data_anon.to(self.device)
        self.train_pairs = train_pairs.to(self.device)
        self.test_pairs = test_pairs.to(self.device)

        print(f"[Data] 数据加载完毕. 节点特征维度: {self.data_orig.x.shape[1]}")

    def _build_model(self):
        """内部方法: 构建模型"""
        input_dim = self.data_orig.x.shape[1]

        self.model = GraphSAGE(
            in_dim=input_dim,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUTPUT_DIM,
            dropout=self.cfg.DROPOUT
        ).to(self.device)

        print(f"[Model] 模型已构建: Input={input_dim}, Hidden={self.cfg.HIDDEN_DIM}, Out={self.cfg.OUTPUT_DIM}")

    def _compute_loss(self, emb_orig, emb_anon):
        """
        计算损失函数
        这里目前使用 MSE Loss (最小化正样本距离)=
        """
        seed_emb_orig = emb_orig[self.train_pairs[:, 0]]
        seed_emb_anon = emb_anon[self.train_pairs[:, 1]]

        loss = F.mse_loss(seed_emb_orig, seed_emb_anon)
        return loss

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        emb_orig, emb_anon = self.model(self.data_orig, self.data_anon)

        loss = self._compute_loss(emb_orig, emb_anon)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self):
        print(f"\n[Train] 开始训练，共 {self.cfg.EPOCHS} 个 Epochs...")
        start_time = time.time()

        try:
            for epoch in range(1, self.cfg.EPOCHS + 1):
                loss = self.train_epoch()

                # 打印日志 (每10轮或第一轮)
                if epoch == 1 or epoch % 10 == 0:
                    print(f"Epoch {epoch:03d}/{self.cfg.EPOCHS} | Loss: {loss:.6f}")

        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")

        end_time = time.time()
        print(f"\n[Train] 训练结束. 总耗时: {end_time - start_time:.2f}s")
        self.save_model()

    def save_model(self):
        """保存模型状态"""
        save_path = self.cfg.MODEL_SAVE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path)
        print(f"💾 模型已保存至: {save_path}")


def main():
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    main()