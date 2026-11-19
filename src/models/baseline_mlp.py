"""Baseline (Core Req. #3): simple 2-layer MLP on precomputed 561-dim handcrafted features."""
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, in_dim=561, hidden_dim=128, num_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):  # x: (B, 561)
        return self.net(x)
