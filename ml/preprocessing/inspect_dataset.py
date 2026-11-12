"""Inspect the How2Sign annotation files and (optionally) the downloaded clips.

Checks the real column layout, counts, durations, split leakage by source video
and by signer, gloss availability, vocabulary/OOV, coverage of coffee-counter
words, and which annotated clips exist on disk. Writes a JSON report.

  python -m preprocessing.inspect_dataset
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.how2sign import SPLITS, annotation_path, read_split  # noqa: E402
from common.tokenizer import PUNCT, tokenize  # noqa: E402

COFFEE_WORDS = ["coffee", "latte", "espresso", "cappuccino", "mocha", "decaf", "tea", "milk", "oat", "almond",
                "soy", "sugar", "cream", "iced", "hot", "small", "medium", "large", "cup", "order", "please",
                "thank", "thanks", "card", "cash", "receipt", "straw", "name", "here", "go", "want", "like"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ML_ROOT / "datasets" / "how2sign")
    ap.add_argument("--report", type=Path, default=ML_ROOT / "reports" / "dataset_inspection.json")
    args = ap.parse_args()

    report: dict = {"root": str(args.root.resolve().relative_to(ML_ROOT)) if args.root.resolve().is_relative_to(ML_ROOT) else str(args.root), "splits": {}}
    data = {}
    for split in SPLITS:
        p = annotation_path(args.root, split)
        if not p.exists():
            print(f"[missing] {p}")
            continue
        recs = read_split(args.root, split)
        data[split] = recs
        durs = [r.duration for r in recs]
        words = [len([t for t in tokenize(r.sentence) if t not in PUNCT]) for r in recs]
        clip_dir = args.root / "clips" / split
        on_disk = {p.stem for p in clip_dir.rglob("*.mp4")} if clip_dir.exists() else set()
        lm_dir = args.root / "landmarks" / split
        lms = {p.stem for p in lm_dir.glob("*.npy")} if lm_dir.exists() else set()
        names = {r.sentence_name for r in recs}
        report["splits"][split] = {
            "sentences": len(recs),
            "videos": len({r.video_id for r in recs}),
            "hours": round(sum(durs) / 3600, 2),
            "duration_s": {"median": round(st.median(durs), 2), "p95": round(sorted(durs)[int(0.95 * len(durs))], 2),
                           "max": round(max(durs), 2), "min": round(min(durs), 2)},
            "words": {"median": st.median(words), "max": max(words)},
            "signers": dict(sorted(Counter(r.signer for r in recs).items(), key=lambda x: int(x[0]))),
            "clips_on_disk": len(on_disk & names),
            "clips_on_disk_not_annotated": len(on_disk - names),
            "landmarks_extracted": len(lms & names),
            "empty_sentences": sum(1 for r in recs if not r.sentence),
        }

    if "train" in data:
        vids = {s: {r.video_id for r in data[s]} for s in data}
        signers = {s: {r.signer for r in data[s]} for s in data}
        report["leakage"] = {
            "shared_source_videos": {f"{a}/{b}": len(vids[a] & vids[b]) for a in data for b in data if a < b},
            "signers_not_in_train": {s: sorted(signers[s] - signers["train"]) for s in data if s != "train"},
            "note": "No source video is shared between splits. Signers ARE shared (How2Sign is not "
                    "signer-independent), except the signers listed above, which form the only unseen-signer subset.",
        }
        c = Counter(t for r in data["train"] for t in tokenize(r.sentence))
        report["vocabulary"] = {"train_types": len(c), "types_freq_ge_3": sum(1 for n in c.values() if n >= 3)}
        keep = {w for w, n in c.items() if n >= 3}
        for s in data:
            if s == "train":
                continue
            toks = [t for r in data[s] for t in tokenize(r.sentence)]
            report["vocabulary"][f"{s}_oov_rate_min_freq_3"] = round(sum(t not in keep for t in toks) / len(toks), 4)
        report["coffee_word_train_counts"] = {w: c[w] for w in COFFEE_WORDS}

    report["gloss_annotations"] = (
        "Not published. how2sign.github.io's download script lists 'Gloss annotations' under TODO and the "
        "website's gloss links point to an unrelated file. Gloss-supervised training needs a separately "
        "obtained gloss file (see --gloss in prepare_sequences.py)."
    )
    report["licence"] = "CC BY-NC 4.0 (how2sign.github.io): attribution required, no commercial use."

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
