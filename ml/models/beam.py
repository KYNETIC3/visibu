"""Beam search with n-gram repetition blocking, returning the n best sentences.

Framework-free: `step(prefixes)` returns next-token log-probs [len(prefixes), V]
as a numpy array, so the same search runs on the PyTorch model (evaluation)
and on the exported ONNX decoder (validation). src/ml/beam.ts is a line-for-line
port; ml/tools/make_parity_fixture.py checks the two agree.

Scoring: a hypothesis's score is its summed log-prob divided by its length
(tokens including <eos>), i.e. its mean token log-prob. exp(score) is the
"model score" shown in the app. It is not a calibrated accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

NEG = -1e9


@dataclass
class BeamConfig:
    beam: int = 4
    n_best: int = 3
    max_len: int = 60
    no_repeat_ngram: int = 3
    banned: Sequence[int] = field(default_factory=tuple)  # e.g. pad, bos, unk


@dataclass
class Hypothesis:
    tokens: list[int]  # without <bos> / <eos>
    score: float  # mean log-prob per token, <eos> included when finished
    finished: bool


def blocked_tokens(tokens: list[int], n: int) -> set[int]:
    """Tokens that would complete an n-gram already present in `tokens`."""
    if n <= 0 or len(tokens) < n - 1:
        return set()
    if n == 1:
        return set(tokens)
    prefix = tuple(tokens[len(tokens) - (n - 1):])
    out = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n - 1]) == prefix:
            out.add(tokens[i + n - 1])
    return out


def beam_search(step: Callable[[list[list[int]]], np.ndarray], bos: int, eos: int,
                cfg: BeamConfig) -> list[Hypothesis]:
    """Returns up to cfg.n_best distinct hypotheses, best first.

    Deterministic: ties are broken by beam index, then token id, so the
    TypeScript port produces the same order given the same log-probs.
    """
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    finished: list[Hypothesis] = []
    for _ in range(cfg.max_len):
        lp = np.asarray(step([[bos] + t for t, _ in beams]), dtype=np.float64)
        cands: list[tuple[float, int, int]] = []  # (total logp, beam idx, token)
        for b, (toks, total) in enumerate(beams):
            row = lp[b].copy()
            for k in cfg.banned:
                row[k] = NEG
            for k in blocked_tokens(toks, cfg.no_repeat_ngram):
                if k != eos:
                    row[k] = NEG
            if not toks:
                row[eos] = NEG  # an empty sentence is never a useful suggestion
            # Top-k per beam, highest first, lower token id first on ties.
            top = np.lexsort((np.arange(len(row)), -row))[: cfg.beam * 2]
            cands += [(total + float(row[k]), b, int(k)) for k in top if row[k] > NEG / 2]
        cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        nxt: list[tuple[list[int], float]] = []
        for total, b, k in cands:
            toks = beams[b][0]
            if k == eos:
                finished.append(Hypothesis(list(toks), total / (len(toks) + 1), True))
            else:
                nxt.append((toks + [k], total))
            if len(nxt) == cfg.beam:
                break
        beams = nxt
        if not beams or len(finished) >= cfg.beam:
            break
    # Finished sentences first; sentences cut off at max_len only fill any remaining slots.
    finished.sort(key=lambda h: -h.score)
    live = sorted((Hypothesis(t, total / max(1, len(t)), False) for t, total in beams), key=lambda h: -h.score)
    pool = finished + live
    out, seen = [], set()
    for h in pool:
        key = tuple(h.tokens)
        if key in seen or not h.tokens:
            continue
        seen.add(key)
        out.append(h)
        if len(out) == cfg.n_best:
            break
    return out
