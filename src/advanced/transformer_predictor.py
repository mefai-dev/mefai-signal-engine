"""
Mefai Signal Engine - Transformer Price Predictor

LSTM + Multi-Head Attention model for cryptocurrency price prediction.
Uses PyTorch with proper training loops, MC Dropout uncertainty estimation,
and cosine annealing learning rate scheduling.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 96
PREDICTION_HORIZON = 12
MC_DROPOUT_PASSES = 50


@dataclass
class PredictionResult:
    predictions: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    mean_prediction: np.ndarray
    upper_bound: np.ndarray
    lower_bound: np.ndarray


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with 8 heads."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        return self.W_o(attn_output)


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feed-forward + layer norm + dropout."""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class PricePredictor(nn.Module):
    """
    LSTM + Transformer price prediction model.

    Architecture:
        Input embedding (linear projection) -> Positional encoding ->
        4 Transformer blocks -> Global average pooling -> Linear head

    Input: (batch, seq_len=96, n_features)
    Output: (batch, prediction_horizon=12)
    """

    def __init__(
        self,
        n_features: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_blocks: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = SEQUENCE_LENGTH,
        pred_horizon: int = PREDICTION_HORIZON,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        self.input_embedding = nn.Linear(n_features, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_horizon),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
               Features: [open, high, low, close, volume]

        Returns:
            Predicted close prices of shape (batch, pred_horizon)
        """
        x = self.input_embedding(x)
        x, _ = self.lstm(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.output_norm(x)
        x = x.mean(dim=1)
        return self.output_head(x)


class PriceDataset(Dataset):
    """Dataset for sliding window price sequences."""

    def __init__(
        self, prices: np.ndarray, seq_len: int = SEQUENCE_LENGTH,
        pred_horizon: int = PREDICTION_HORIZON
    ):
        """
        Args:
            prices: Array of shape (n_candles, n_features).
                    Features: [open, high, low, close, volume]
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        self.mean = prices.mean(axis=0)
        self.std = prices.std(axis=0) + 1e-8
        self.data = (prices - self.mean) / self.std

        self.close_idx = 3
        self.n_samples = len(prices) - seq_len - pred_horizon + 1

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_horizon, self.close_idx
        ]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class PricePredictorTrainer:
    """Training and inference wrapper for PricePredictor."""

    def __init__(
        self,
        n_features: int = 5,
        d_model: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = PricePredictor(n_features=n_features, d_model=d_model).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        self.scheduler = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.normalization_params: Optional[Dict] = None

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict:
        """
        Train the model with early stopping.

        Args:
            train_data: Training candles, shape (n_candles, n_features)
            val_data: Validation candles (if None, uses last 20% of train_data)
            epochs: Maximum training epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Training history dict
        """
        if val_data is None:
            split = int(len(train_data) * 0.8)
            val_data = train_data[split:]
            train_data = train_data[:split]

        train_dataset = PriceDataset(train_data)
        val_dataset = PriceDataset(val_data)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                f"Not enough data. Need at least {SEQUENCE_LENGTH + PREDICTION_HORIZON} candles. "
                f"Got train={len(train_data)}, val={len(val_data)}"
            )

        self.normalization_params = {
            "mean": train_dataset.mean.tolist(),
            "std": train_dataset.std.tolist(),
        }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - "
                f"LR: {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
            "device": str(self.device),
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                predictions = self.model(x_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict(
        self, candles: np.ndarray, n_passes: int = MC_DROPOUT_PASSES
    ) -> PredictionResult:
        """
        Predict next candles with uncertainty estimation via MC Dropout.

        Args:
            candles: Recent candles, shape (seq_len, n_features)
            n_passes: Number of MC Dropout forward passes

        Returns:
            PredictionResult with predictions, uncertainty, and confidence
        """
        if self.normalization_params is None:
            raise RuntimeError("Model must be trained before prediction")

        mean = np.array(self.normalization_params["mean"])
        std = np.array(self.normalization_params["std"])

        normalized = (candles - mean) / std
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.train()
        all_predictions = []

        with torch.no_grad():
            for _ in range(n_passes):
                pred = self.model(x).cpu().numpy()[0]
                pred_denorm = pred * std[3] + mean[3]
                all_predictions.append(pred_denorm)

        self.model.eval()

        all_predictions = np.array(all_predictions)
        mean_pred = all_predictions.mean(axis=0)
        std_pred = all_predictions.std(axis=0)

        avg_uncertainty = std_pred.mean()
        avg_price = np.abs(mean_pred).mean()
        relative_uncertainty = avg_uncertainty / max(avg_price, 1e-8)
        confidence = max(0.0, min(1.0, 1.0 - relative_uncertainty * 10))

        return PredictionResult(
            predictions=all_predictions,
            uncertainty=std_pred,
            confidence=confidence,
            mean_prediction=mean_pred,
            upper_bound=mean_pred + 2 * std_pred,
            lower_bound=mean_pred - 2 * std_pred,
        )

    def save(self, path: str):
        """Save model weights and normalization parameters."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "normalization_params": self.normalization_params,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights and normalization parameters."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.normalization_params = checkpoint["normalization_params"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Model loaded from {path}")
