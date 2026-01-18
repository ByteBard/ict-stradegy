"""
LSTM分类器模型

用于预测交易信号质量
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LSTMConfig:
    """LSTM模型配置"""
    input_dim: int = 48          # 输入特征维度
    hidden_dim: int = 64         # LSTM隐藏层维度
    num_layers: int = 2          # LSTM层数
    dropout: float = 0.3         # Dropout比例
    bidirectional: bool = False  # 是否双向LSTM
    fc_dim: int = 32             # 全连接层维度
    output_dim: int = 1          # 输出维度 (1=二分类概率)


class LSTMClassifier(nn.Module):
    """
    LSTM分类器

    架构:
        Input (seq_len, input_dim)
            ↓
        LSTM层 (num_layers, hidden_dim)
            ↓
        Dropout
            ↓
        FC层 (fc_dim)
            ↓
        ReLU
            ↓
        Dropout
            ↓
        Output层 (output_dim)
            ↓
        Sigmoid (二分类)
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        self.config = config or LSTMConfig()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional,
        )

        # 计算LSTM输出维度
        lstm_output_dim = self.config.hidden_dim
        if self.config.bidirectional:
            lstm_output_dim *= 2

        # 全连接层
        self.fc1 = nn.Linear(lstm_output_dim, self.config.fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc2 = nn.Linear(self.config.fc_dim, self.config.output_dim)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1，帮助长期记忆
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            hidden: 可选的初始隐藏状态

        Returns:
            输出概率 (batch, output_dim)
        """
        # LSTM层
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # 取最后一个时间步的输出
        # lstm_out: (batch, seq_len, hidden_dim)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测接口 (numpy输入输出)

        Args:
            x: 输入特征 (seq_len, input_dim) 或 (batch, seq_len, input_dim)

        Returns:
            预测概率
        """
        self.eval()

        # 确保是3D输入
        if x.ndim == 2:
            x = x[np.newaxis, :]

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            prob = self.forward(x_tensor)
            return prob.numpy().flatten()

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> 'LSTMClassifier':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


class LSTMTrainer:
    """
    LSTM训练器

    提供训练、验证、早停等功能
    """

    def __init__(
        self,
        model: LSTMClassifier,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

                # 计算准确率
                predicted = (outputs >= 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0

        self.val_losses.append(avg_loss)
        self.scheduler.step(avg_loss)

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 15,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 最大epoch数
            patience: 早停耐心值
            save_path: 模型保存路径

        Returns:
            训练历史记录
        """
        print(f"开始训练 (最大{epochs}轮, 早停耐心={patience})")
        print("-" * 50)

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 打印进度
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.2%}")

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 保存最佳模型
                if save_path:
                    self.model.save(save_path)
                    print(f"  → 保存最佳模型 (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n早停触发 (连续{patience}轮无改进)")
                    break

        print("-" * 50)
        print(f"训练完成! 最佳验证损失: {self.best_val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
        }
