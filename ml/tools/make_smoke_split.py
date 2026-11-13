"""Carve a SMOKE-TEST train/eval pair out of one prepared manifest, split by source video.

Only for checking that the pipeline runs end to end on real How2Sign data before
the full training set is available. The official test split is never touched.
Results from smoke runs measure nothing about translation quality.

  python -m tools.make_smoke_split --source val --eval-fraction 0.2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from training.data import load_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared")
    ap.add_argument("--source", default="val")
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.source == "test":
        sys.exit("Never carve smoke splits from the test set.")
    rows = load_manifest(args.prepared / f"{args.source}.jsonl")
    videos = sorted({r["video_id"] for r in rows})
    random.Random(args.seed).shuffle(videos)
    held = set(videos[: max(1, int(len(videos) * args.eval_fraction))])
    for name, part in (("smoke_train", [r for r in rows if r["video_id"] not in held]),
                       ("smoke_eval", [r for r in rows if r["video_id"] in held])):
        with open(args.prepared / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
        print(name, len(part), "examples")


if __name__ == "__main__":
    main()
