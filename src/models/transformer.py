"""
Modern model (Core Req. #4b): Transformer encoder, trained from scratch.

Fulfills Core Req. #5 ("write at least one key component yourselves rather than
calling it from a library"): multi-head self-attention is implemented manually
below (ManualMultiHeadSelfAttention) instead of using nn.MultiheadAttention or
nn.TransformerEncoderLayer.

Note: this model is trained from scratch, not pretrained, per project scoping
(see README / technical report, Core Req. #7 pretrained-vs-scratch study marked N/A).
"""
import math
import torch
import torch.nn as nn


class ManualMultiHeadSelfAttention(nn.Module):
    """Hand-written scaled dot-product multi-head self-attention (no nn.MultiheadAttention)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B,H,T,dh)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,T,dh)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = ManualMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, in_channels=9, d_model=64, num_heads=4, num_layers=3,
                 ff_dim=128, num_classes=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: (B, 128, C)
        B = x.size(0)
        x = self.input_proj(x)  # (B, T, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token representation
        return self.head(cls_out)
