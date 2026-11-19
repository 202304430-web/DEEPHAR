"""Classic model (Core Req. #4a): bidirectional LSTM over raw IMU windows."""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):  # x: (B, 128, C)
        _, (h_n, _) = self.lstm(x)
        h_fwd, h_bwd = h_n[-2], h_n[-1]  # final-layer fwd/bwd hidden states
        feat = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.head(feat)
