"""Streaming dataset over prepared manifests. Landmark files are memory-mapped
per example and features are computed on the fly, so the dataset is never held
in memory."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import sequence_features  # noqa: E402
from common.layout import HAND_POINTS, LEFT_START, RIGHT_START  # noqa: E402
from common.tokenizer import PAD_ID, Vocab  # noqa: E402


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def augment(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """Training-only augmentation on raw packed frames [T, 553, 3]."""
    T = len(frames)
    # Signing speed: resample in time by 0.8x..1.25x.
    speed = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
    n = max(2, int(round(T / speed)))
    idx = np.clip(np.round(np.linspace(0, T - 1, n)).astype(int), 0, T - 1)
    f = frames[idx].copy()
    # Camera: small rotation, per-axis scale, shift (image coordinates).
    ang = math.radians(rng.uniform(-8, 8))
    sx, sy = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    dx, dy = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
    x, y = f[..., 0] - 0.5, f[..., 1] - 0.5
    c, s = math.cos(ang), math.sin(ang)
    f[..., 0] = (c * x - s * y) * sx + 0.5 + dx
    f[..., 1] = (s * x + c * y) * sy + 0.5 + dy
    # Occlusion: hide one hand for a span of frames.
    if rng.random() < 0.15 and len(f) > 4:
        start = LEFT_START if rng.random() < 0.5 else RIGHT_START
        a = rng.randrange(len(f))
        b = min(len(f), a + rng.randrange(1, max(2, len(f) // 4)))
        f[a:b, start:start + HAND_POINTS] = np.nan
    return f


class SignTextDataset(Dataset):
    def __init__(self, rows: list[dict], landmarks_root: Path, vocab: Vocab, stats: dict | None = None,
                 train: bool = False, seed: int = 0, ctc_source: str = "auto"):
        self.rows = rows
        self.root = Path(landmarks_root)
        self.vocab = vocab
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.ctc_source = ctc_source

    def __len__(self) -> int:
        return len(self.rows)

    def ctc_ids(self, r: dict) -> list[int]:
        use_gloss = self.ctc_source == "gloss" or (self.ctc_source == "auto" and r.get("gloss"))
        if use_gloss:
            return [self.vocab.stoi.get(g, 3) for g in r["gloss"].split()]
        return self.vocab.ctc_targets(r["sentence"])

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        frames = np.load(self.root / r["landmarks"], mmap_mode="r").astype(np.float32)
        if self.train:
            frames = augment(frames, random.Random(hash((self.seed, self.epoch, i))))
        feats = sequence_features(frames, r["aspect"])
        return {
            "feats": torch.from_numpy(feats),
            "tokens": torch.tensor(self.vocab.encode(r["sentence"]), dtype=torch.long),
            "ctc": torch.tensor(self.ctc_ids(r), dtype=torch.long),
            "name": r["name"], "signer": r["signer"], "sentence": r["sentence"],
        }


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["feats"].shape[0] for b in batch)
    F = batch[0]["feats"].shape[1]
    L = max(len(b["tokens"]) for b in batch)
    feats = torch.zeros(B, T, F)
    toks = torch.full((B, L), PAD_ID, dtype=torch.long)
    for j, b in enumerate(batch):
        feats[j, : len(b["feats"])] = b["feats"]
        toks[j, : len(b["tokens"])] = b["tokens"]
    return {
        "feats": feats,
        "feat_lens": torch.tensor([len(b["feats"]) for b in batch]),
        "tokens_in": toks[:, :-1],
        "tokens_out": toks[:, 1:],
        "ctc": torch.cat([b["ctc"] for b in batch]) if any(len(b["ctc"]) for b in batch) else torch.zeros(0, dtype=torch.long),
        "ctc_lens": torch.tensor([len(b["ctc"]) for b in batch]),
        "names": [b["name"] for b in batch],
        "signers": [b["signer"] for b in batch],
        "sentences": [b["sentence"] for b in batch],
    }


class BucketSampler(Sampler):
    """Batches of similar length (less padding), shuffled each epoch."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.batch_size * 50
        batches = []
        for c in range(0, len(idx), chunk):
            part = sorted(idx[c:c + chunk], key=lambda i: self.lengths[i])
            batches += [part[k:k + self.batch_size] for k in range(0, len(part), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)
