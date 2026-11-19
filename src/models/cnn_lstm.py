"""
Advanced architecture (Core Req. #4c): CNN-LSTM hybrid.
1D-CNN stack extracts local temporal-spatial features across sensor channels,
a BiLSTM models longer-range temporal dependencies over the CNN feature maps,
and a learned additive-attention pooling layer (not library-provided) aggregates
the sequence into a single vector before classification.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool else nn.Identity()

    def forward(self, x):  # x: (B, C, T)
        return self.pool(self.act(self.bn(self.conv(x))))


class AdditiveAttentionPool(nn.Module):
    """Learned attention pooling over the time axis (hand-written, not a library call)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, T, D)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, weights


class CNNLSTMHybrid(nn.Module):
    def __init__(self, in_channels=9, cnn_channels=(32, 64), lstm_hidden=64,
                 num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, cnn_channels[0], kernel_size=5, pool=2)
        self.conv2 = ConvBlock(cnn_channels[0], cnn_channels[1], kernel_size=3, pool=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = AdditiveAttentionPool(lstm_hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x):  # x: (B, 128, C)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.conv1(x)
        x = self.conv2(x)              # (B, C', T')
        x = x.transpose(1, 2)          # (B, T', C')
        seq, _ = self.lstm(x)          # (B, T', 2*hidden)
        pooled, _ = self.attn_pool(seq)
        return self.head(pooled)
