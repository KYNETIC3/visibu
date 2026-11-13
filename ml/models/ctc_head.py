"""CTC output head and greedy CTC decoding.

With gloss annotations the head predicts glosses (continuous sign recognition).
How2Sign publishes no glosses, so by default it is trained on the English words
of the sentence as an AUXILIARY alignment loss. Those outputs are not glosses
and are never shown to the user as a recognition result.
"""
from __future__ import annotations

import torch
from torch import nn


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory [B, T', D] -> log-probabilities [B, T', V]."""
        return self.proj(memory).log_softmax(-1)


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor | None = None, blank: int = 0) -> list[list[int]]:
    """Best path: argmax per frame, collapse repeats, drop blanks. Mirrored by src/ml/ctc.ts."""
    best = log_probs.argmax(-1).cpu().tolist()
    lens = lengths.cpu().tolist() if lengths is not None else [len(b) for b in best]
    out = []
    for seq, n in zip(best, lens):
        ids, prev = [], None
        for t in seq[:n]:
            if t != prev and t != blank:
                ids.append(t)
            prev = t
        out.append(ids)
    return out
