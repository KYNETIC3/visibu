"""Compute per-feature mean and standard deviation over the TRAIN split.

The geometric normalisation (shoulder-relative, wrist-relative, face-relative)
happens in common/features.py, for every frame, in Python and the browser. This
script adds a second, statistical standardisation: the statistics are stored in
the checkpoint and baked into the exported ONNX encoder, so the browser never
needs to apply them itself. Presence flags are left unscaled.

Streams over files (Welford-style sums), never loading the dataset into memory.

  python -m preprocessing.normalize_landmarks [--max-examples 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_DIM, sequence_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared")
    ap.add_argument("--max-examples", type=int, default=5000)
    ap.add_argument("--manifest", default="train", help="statistics come from this manifest only (never val/test)")
    args = ap.parse_args()

    cfg = json.loads((args.prepared / "prep_config.json").read_text())
    lm_root = Path(cfg["landmarks_root"])
    rows = [json.loads(l) for l in open(args.prepared / f"{args.manifest}.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(0)
    if len(rows) > args.max_examples:
        rows = [rows[i] for i in rng.choice(len(rows), args.max_examples, replace=False)]

    n = 0
    s1 = np.zeros(FEATURE_DIM)
    s2 = np.zeros(FEATURE_DIM)
    for i, r in enumerate(rows):
        x = sequence_features(np.load(lm_root / r["landmarks"]).astype(np.float32), r["aspect"]).astype(np.float64)
        n += len(x)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean ** 2, 0.0))
    std = np.where(std < 1e-3, 1.0, std)
    mean[-4:] = 0.0  # presence flags stay 0/1
    std[-4:] = 1.0
    (args.prepared / "feature_stats.json").write_text(json.dumps({
        "frames": int(n), "examples": len(rows), "manifest": args.manifest, "mean": mean.tolist(), "std": std.tolist(),
    }))
    print(f"wrote feature_stats.json from {len(rows)} examples / {n} frames")


if __name__ == "__main__":
    main()
