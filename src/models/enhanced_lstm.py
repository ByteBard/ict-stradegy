"""
增强版LSTM模型

改进:
1. 多任务输出 - 同时预测方向和幅度
2. Attention机制 - 关注重要时间步
3. 残差连接 - 更好的梯度传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnhancedLSTMConfig:
    """增强LSTM配置"""
    input_dim: int = 48
    hidden_dim: int = 128          # 增大隐藏层
    num_layers: int = 2
    dropout: float = 0.3
    use_attention: bool = True     # 使用注意力
    num_heads: int = 4             # 多头注意力

    # 多任务输出
    output_direction: bool = True   # 预测方向
    output_magnitude: bool = True   # 预测幅度
    output_confidence: bool = True  # 预测置信度


class TemporalAttention(nn.Module):
    """时序注意力层"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class EnhancedLSTM(nn.Module):
    """
    增强版LSTM

    架构:
        Input (seq_len, input_dim)
            ↓
        Input Projection (hidden_dim)
            ↓
        LSTM层 x num_layers
            ↓
        Temporal Attention (可选)
            ↓
        全连接层
            ↓
        多任务输出:
          - direction: P(做空盈利) ∈ [0,1]
          - magnitude: 预期收益幅度
          - confidence: 预测置信度 ∈ [0,1]
    """

    def __init__(self, config: Optional[EnhancedLSTMConfig] = None):
        super().__init__()
        self.config = config or EnhancedLSTMConfig()

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=False
        )

        # 注意力层
        if self.config.use_attention:
            self.attention = TemporalAttention(
                self.config.hidden_dim,
                self.config.num_heads
            )
        else:
            self.attention = None

        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        # 多任务输出
        fc_out_dim = self.config.hidden_dim // 2

        if self.config.output_direction:
            self.direction_head = nn.Sequential(
                nn.Linear(fc_out_dim, 1),
                nn.Sigmoid()
            )

        if self.config.output_magnitude:
            self.magnitude_head = nn.Linear(fc_out_dim, 1)

        if self.config.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(fc_out_dim, 1),
                nn.Sigmoid()
            )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        前向传播

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            dict with keys: direction, magnitude, confidence
        """
        # 输入投影
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # 注意力
        if self.attention is not None:
            lstm_out = self.attention(lstm_out)

        # 取最后时间步
        last_out = lstm_out[:, -1, :]

        # 共享特征
        shared = self.fc(last_out)

        # 多任务输出
        outputs = {}

        if self.config.output_direction:
            outputs['direction'] = self.direction_head(shared)

        if self.config.output_magnitude:
            outputs['magnitude'] = self.magnitude_head(shared)

        if self.config.output_confidence:
            outputs['confidence'] = self.confidence_head(shared)

        return outputs

    def predict(self, x: np.ndarray) -> dict:
        """预测接口"""
        self.eval()

        if x.ndim == 2:
            x = x[np.newaxis, :]

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            outputs = self.forward(x_tensor)

            result = {}
            for k, v in outputs.items():
                result[k] = v.numpy().flatten()

            return result

    def save(self, path: str):
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str) -> 'EnhancedLSTM':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(
        self,
        direction_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        confidence_weight: float = 0.3
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.confidence_weight = confidence_weight

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        outputs: dict,
        targets: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算多任务损失

        Args:
            outputs: 模型输出 {direction, magnitude, confidence}
            targets: 目标值 {direction, magnitude}

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}
        total = 0

        # 方向损失 (BCE)
        if 'direction' in outputs and 'direction' in targets:
            loss_dir = self.bce(outputs['direction'], targets['direction'])
            losses['direction'] = loss_dir.item()
            total += self.direction_weight * loss_dir

        # 幅度损失 (MSE)
        if 'magnitude' in outputs and 'magnitude' in targets:
            loss_mag = self.mse(outputs['magnitude'], targets['magnitude'])
            losses['magnitude'] = loss_mag.item()
            total += self.magnitude_weight * loss_mag

        # 置信度损失 - 用预测是否正确作为目标
        if 'confidence' in outputs and 'direction' in targets:
            # 预测正确时置信度应该高
            with torch.no_grad():
                pred_correct = (
                    (outputs['direction'] >= 0.5) == (targets['direction'] >= 0.5)
                ).float()
            loss_conf = self.bce(outputs['confidence'], pred_correct)
            losses['confidence'] = loss_conf.item()
            total += self.confidence_weight * loss_conf

        return total, losses


class EnhancedLSTMTrainer:
    """增强LSTM训练器"""

    def __init__(
        self,
        model: EnhancedLSTM,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.criterion = MultiTaskLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader) -> dict:
        self.model.train()
        total_loss = 0
        loss_components = {'direction': 0, 'magnitude': 0, 'confidence': 0}
        n_batches = 0

        for batch_x, batch_y_dir, batch_y_mag in train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(batch_x)
            targets = {
                'direction': batch_y_dir,
                'magnitude': batch_y_mag
            }

            loss, losses = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in losses.items():
                loss_components[k] += v
            n_batches += 1

        self.scheduler.step()

        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}

        return {'total': avg_loss, **avg_components}

    def validate(self, val_loader) -> Tuple[dict, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y_dir, batch_y_mag in val_loader:
                outputs = self.model(batch_x)
                targets = {
                    'direction': batch_y_dir,
                    'magnitude': batch_y_mag
                }

                loss, _ = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 方向准确率
                pred = (outputs['direction'] >= 0.5).float()
                correct += (pred == batch_y_dir).sum().item()
                total += batch_y_dir.size(0)
                n_batches += 1

        avg_loss = total_loss / n_batches
        accuracy = correct / total if total > 0 else 0

        return {'total': avg_loss}, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: int = 15,
        save_path: Optional[str] = None
    ) -> dict:
        print(f"开始训练 (最大{epochs}轮, 早停={patience})")
        print("-" * 60)

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss['total'])
            self.val_losses.append(val_loss['total'])

            print(f"Epoch {epoch+1:3d}: "
                  f"Train={train_loss['total']:.4f} "
                  f"(D:{train_loss['direction']:.3f} M:{train_loss['magnitude']:.3f}) "
                  f"Val={val_loss['total']:.4f} Acc={val_acc:.1%}")

            if val_loss['total'] < self.best_val_loss:
                self.best_val_loss = val_loss['total']
                patience_counter = 0
                if save_path:
                    self.model.save(save_path)
                    print(f"  → 保存最佳模型")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发")
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
