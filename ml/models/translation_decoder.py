"""Model B, translation: Transformer decoder that writes English from the sign encoder's memory.

Autoregressive, teacher-forced in training, greedy or beam search at inference.
Output embedding is tied to the input embedding.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .sign_encoder import sinusoidal


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, layers: int = 3, heads: int = 4, ff: int = 1024,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        # Tied weights: N(0, d^-0.5) keeps initial logits small (default N(0, 1) gave
        # an initial loss of ~118 instead of ~ln(vocab) = 8.9).
        nn.init.normal_(self.embed.weight, 0.0, d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()
        self.d_model = d_model
        self.pad_id = pad_id
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor | None = None):
        """tokens [B, L] (starting with <bos>), memory [B, S, D] -> logits [B, L, V]."""
        L = tokens.size(1)
        h = self.embed(tokens) * math.sqrt(self.d_model) + sinusoidal(L, self.d_model, tokens.device)
        h = self.drop(h)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
        tgt_pad = tokens == self.pad_id if self.training else None
        # tgt_is_causal=True: skips PyTorch's data-dependent mask check, which ONNX export cannot trace.
        h = self.decoder(h, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_pad,
                         memory_key_padding_mask=memory_pad_mask, tgt_is_causal=True)
        return self.out(self.norm(h))
