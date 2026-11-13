"""Model A, recognition: landmark features -> temporal Transformer encoder.

features [B, T, F]
  -> standardise with train mean/std (buffers, baked into ONNX)
  -> Linear + LayerNorm + GELU            per-frame embedding
  -> Conv1d(k=5, stride=2) x subsample    local motion, halves the frame rate
  -> + sinusoidal position encoding       order of movements
  -> TransformerEncoder (pre-norm)        sentence-level context
  -> memory [B, T', D]
"""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SignEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, layers: int = 4, heads: int = 4, ff: int = 1024,
                 dropout: float = 0.1, subsample: int = 2):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.embed = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.subsample = subsample
        self.conv = nn.ModuleList(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2) for _ in range(int(math.log2(subsample)))
        )
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def set_stats(self, mean, std) -> None:
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for _ in self.conv:
            lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        return lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """x [B, T, F]; lengths [B] or None (no padding). Returns memory [B, T', D], pad mask [B, T'] or None."""
        x = (x - self.feat_mean) / self.feat_std
        h = self.embed(x)
        if lengths is not None:
            frame_mask = torch.arange(h.size(1), device=h.device)[None, :] < lengths[:, None]
            h = h * frame_mask.unsqueeze(-1)
        h = h.transpose(1, 2)
        for conv in self.conv:
            h = torch.nn.functional.gelu(conv(h))
        h = h.transpose(1, 2)
        h = h * math.sqrt(self.d_model) + sinusoidal(h.size(1), self.d_model, h.device)
        pad_mask = None
        if lengths is not None:
            out_len = self.out_lengths(lengths)
            pad_mask = torch.arange(h.size(1), device=h.device)[None, :] >= out_len[:, None]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.norm(h), pad_mask
