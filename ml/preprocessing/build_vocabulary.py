"""Build the English vocabulary from the TRAIN split only.

  python -m preprocessing.build_vocabulary --min-freq 3 --max-size 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.how2sign import read_split  # noqa: E402
from common.tokenizer import Vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ML_ROOT / "datasets" / "how2sign")
    ap.add_argument("--out", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared" / "vocab.json")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-size", type=int, default=8000)
    ap.add_argument("--only-extracted", action="store_true",
                    help="use only train sentences whose landmarks exist (when only part of train was downloaded)")
    args = ap.parse_args()

    train = read_split(args.root, "train")
    if args.only_extracted:
        lm = args.root / "landmarks" / "train"
        train = [r for r in train if (lm / f"{r.sentence_name}.npy").exists()]
    vocab = Vocab.build((r.sentence for r in train), args.min_freq, args.max_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.out)
    print(json.dumps({"size": len(vocab), "min_freq": args.min_freq, "sentences": len(train), "out": str(args.out)}))


if __name__ == "__main__":
    main()
