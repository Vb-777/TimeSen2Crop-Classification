"""Time-series classifiers. Input: (batch, T=12 months, C channels) → logits (batch, n_classes).

* TempCNN      – Pelletier et al., "Temporal Convolutional Neural Network for the
                 Classification of Satellite Image Time Series", Remote Sensing 2019.
* LSTM         – stacked LSTM over the monthly sequence; the "weighted LSTM" of the
                 TimeSen2Crop benchmark is the same network trained with a
                 class-weighted loss (see train.py).
* Transformer  – self-attention encoder over the months with positional encoding
                 (in the spirit of Rußwurm & Körner, 2020).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TempCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, seq_len: int = 12, filters: int = 64,
                 kernel: int = 5, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(3):
            layers += [nn.Conv1d(c, filters, kernel, padding=kernel // 2), nn.BatchNorm1d(filters), nn.ReLU(),
                       nn.Dropout(dropout)]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * seq_len, hidden), nn.BatchNorm1d(hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, C)
        return self.head(self.conv(x.transpose(1, 2)))


class LSTMClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out), nn.Dropout(dropout), nn.Linear(out, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)
        return self.head(seq[:, -1])  # state after the last month (August)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, d_model: int = 64, heads: int = 4, layers: int = 3,
                 ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.embed(x)))
        return self.head(h.mean(dim=1))  # average over the 12 months


def build(name: str, in_channels: int, n_classes: int, seq_len: int = 12) -> nn.Module:
    name = name.lower()
    if name == "tempcnn":
        return TempCNN(in_channels, n_classes, seq_len)
    if name in ("lstm", "weighted-lstm"):
        return LSTMClassifier(in_channels, n_classes)
    if name == "transformer":
        return TransformerClassifier(in_channels, n_classes)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
