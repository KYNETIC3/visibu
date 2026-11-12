"""Join annotations with extracted landmark files into per-split manifests.

Writes <prepared>/<split>.jsonl with one example per line:
  {name, split, signer, video_id, landmarks, aspect, frames, sentence, gloss}
The original How2Sign split is preserved. Records without a landmark file, with
too few frames, or with an empty sentence are dropped and counted.
Also writes <prepared>/prep_config.json (settings, counts, annotation sha256).

  python -m preprocessing.prepare_sequences [--gloss path/to/gloss.tsv]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_VERSION, TARGET_FPS  # noqa: E402
from common.how2sign import SPLITS, annotation_path, read_glosses, read_split  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ML_ROOT / "datasets" / "how2sign")
    ap.add_argument("--landmarks", type=Path, help="default: <root>/landmarks")
    ap.add_argument("--out", type=Path, help="default: <root>/prepared")
    ap.add_argument("--gloss", type=Path, help="optional TSV with SENTENCE_NAME and GLOSS columns")
    ap.add_argument("--min-frames", type=int, default=8)
    args = ap.parse_args()

    lm_root = args.landmarks or args.root / "landmarks"
    out = args.out or args.root / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    glosses = read_glosses(args.gloss)

    config = {"feature_version": FEATURE_VERSION, "target_fps": TARGET_FPS, "min_frames": args.min_frames,
              "gloss_file": str(args.gloss) if args.gloss else None, "splits": {}, "annotation_sha256": {}}
    for split in SPLITS:
        ann = annotation_path(args.root, split)
        if not ann.exists():
            continue
        config["annotation_sha256"][split] = sha256(ann)
        dropped = {"no_landmarks": 0, "too_short": 0, "empty_sentence": 0, "no_gloss": 0}
        kept = 0
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in read_split(args.root, split):
                npy = lm_root / split / f"{r.sentence_name}.npy"
                meta = npy.with_suffix(".json")
                if not npy.exists() or not meta.exists():
                    dropped["no_landmarks"] += 1
                    continue
                m = json.loads(meta.read_text())
                if m["frames"] < args.min_frames:
                    dropped["too_short"] += 1
                    continue
                if not r.sentence:
                    dropped["empty_sentence"] += 1
                    continue
                gloss = glosses.get(r.sentence_name)
                if glosses and gloss is None:
                    dropped["no_gloss"] += 1
                    continue
                fh.write(json.dumps({
                    "name": r.sentence_name, "split": split, "signer": r.signer, "video_id": r.video_id,
                    "landmarks": str(npy.relative_to(lm_root)), "aspect": m["width"] / m["height"],
                    "frames": m["frames"], "sentence": r.sentence, "gloss": gloss,
                }) + "\n")
                kept += 1
        config["splits"][split] = {"kept": kept, "dropped": dropped}
        print(split, kept, dropped)
    config["landmarks_root"] = str(lm_root)
    (out / "prep_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
