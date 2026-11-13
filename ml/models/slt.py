"""Sign-language translation model: SignEncoder (+ CTC head) and TranslationDecoder.

Trained jointly:  loss = CE(decoder) + ctc_weight * CTC(encoder).
Exported as two ONNX graphs (recognition encoder, translation decoder).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .beam import BeamConfig, Hypothesis, beam_search
from .ctc_head import CTCHead
from .sign_encoder import SignEncoder
from .translation_decoder import TranslationDecoder


@dataclass
class ModelConfig:
    in_dim: int
    vocab_size: int
    d_model: int = 256
    enc_layers: int = 6
    dec_layers: int = 3
    heads: int = 4
    ff: int = 1024
    dropout: float = 0.1
    subsample: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class SLTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SignEncoder(cfg.in_dim, cfg.d_model, cfg.enc_layers, cfg.heads, cfg.ff, cfg.dropout,
                                   cfg.subsample)
        self.ctc = CTCHead(cfg.d_model, cfg.vocab_size)
        self.decoder = TranslationDecoder(cfg.vocab_size, cfg.d_model, cfg.dec_layers, cfg.heads, cfg.ff,
                                          cfg.dropout)

    def forward(self, feats, feat_lens, tokens_in):
        memory, pad = self.encoder(feats, feat_lens)
        return self.ctc(memory), self.encoder.out_lengths(feat_lens), self.decoder(tokens_in, memory, pad)

    @torch.no_grad()
    def greedy(self, feats, feat_lens, bos: int, eos: int, max_len: int = 60):
        """Batched greedy decoding. Returns (token ids per example, mean token prob per example)."""
        memory, pad = self.encoder(feats, feat_lens)
        B = feats.size(0)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=feats.device)
        done = torch.zeros(B, dtype=torch.bool, device=feats.device)
        logp_sum = torch.zeros(B, device=feats.device)
        n_tok = torch.zeros(B, device=feats.device)
        for _ in range(max_len):
            logits = self.decoder(ys, memory, pad)[:, -1]
            lp = logits.log_softmax(-1)
            nxt = lp.argmax(-1)
            step_lp = lp.gather(1, nxt[:, None]).squeeze(1)
            logp_sum += torch.where(done, 0.0, step_lp)
            n_tok += (~done).float()
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if bool(done.all()):
                break
        conf = torch.exp(logp_sum / n_tok.clamp(min=1))
        out = []
        for row in ys[:, 1:].tolist():
            out.append(row[: row.index(eos)] if eos in row else row)
        return out, conf.tolist(), memory, pad

    @torch.no_grad()
    def beam(self, memory: torch.Tensor, bos: int, eos: int, cfg: BeamConfig) -> list[Hypothesis]:
        """Beam search for ONE example. memory [1, S, D] from self.encoder (unpadded)."""

        def step(prefixes: list[list[int]]):
            toks = torch.tensor(prefixes, dtype=torch.long, device=memory.device)
            mem = memory.expand(len(prefixes), -1, -1)
            return self.decoder(toks, mem, None)[:, -1].float().log_softmax(-1).cpu().numpy()

        return beam_search(step, bos, eos, cfg)
