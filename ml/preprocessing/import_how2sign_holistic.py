"""Import "How2Sign Holistic" (pre-extracted MediaPipe Holistic landmarks for every
How2Sign sentence clip) into Visibu's landmark layout, so the full How2Sign
corpus can be trained on without downloading the videos or running MediaPipe.

Source: https://huggingface.co/datasets/PSewmuthu/How2Sign_Holistic, also on Kaggle
as psewmuthu/how2sign-holistic. It is derived from How2Sign, so How2Sign's
CC BY-NC 4.0 terms still apply whatever licence the copy is labelled with.

Format, verified against this repo's own extraction of the same val clips
(ml/README.md, "Full How2Sign from pre-extracted landmarks"):
  <split>/frontal/<SENTENCE_NAME>_holistic.npy   float32 [T, 543, 3]
  points: pose 0..32, face 33..500 (468, no iris), left hand 501..521, right hand 522..542
  T is every frame of the clip at the video's 24 fps; a part that was not detected is all zeros.
Here: zeros -> NaN, the 10 iris slots stay NaN (the features use face points <= 454),
resampled to 15 fps by nearest frame as extract_landmarks.py does.

Writes the same files as extract_landmarks.py, so prepare_sequences.py,
build_vocabulary.py and normalize_landmarks.py work unchanged:
  <out>/annotations/how2sign_realigned_<split>.csv   (copied)
  <out>/landmarks/<split>/<SENTENCE_NAME>.npy        float16 [T, 553, 3]
  <out>/landmarks/<split>/<SENTENCE_NAME>.json       frames, fps, width, height

  python -m preprocessing.import_how2sign_holistic --src /kaggle/input/how2sign-holistic/how2sign_holistic_features --out /tmp/h2s
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import TARGET_FPS  # noqa: E402
from common.how2sign import SPLITS, read_split  # noqa: E402
from common.layout import FACE_START, LEFT_START, POSE_START, RIGHT_START, TOTAL_POINTS  # noqa: E402
from preprocessing.import_asl_signs import resample  # noqa: E402

SOURCE_FPS = 24.0
WIDTH, HEIGHT = 1280, 720  # How2Sign front clips
# (start in the source array, start in ours, points)
PARTS = [(0, POSE_START, 33), (33, FACE_START, 468), (501, LEFT_START, 21), (522, RIGHT_START, 21)]
SOURCE_POINTS = 543


def holistic_to_frames(src: np.ndarray, src_fps: float = SOURCE_FPS) -> np.ndarray:
    """[T, 543, 3] Holistic array -> [T', 553, 3] float32 in Visibu's layout at 15 fps."""
    if src.ndim != 3 or src.shape[1:] != (SOURCE_POINTS, 3):
        raise ValueError(f"expected [T, {SOURCE_POINTS}, 3], got {src.shape}")
    out = np.full((len(src), TOTAL_POINTS, 3), np.nan, dtype=np.float32)
    for s, d, n in PARTS:
        part = src[:, s:s + n].astype(np.float32)
        found = ~np.all(part[..., :2] == 0, axis=(1, 2))
        out[found, d:d + n] = part[found]
    return resample(out, src_fps, TARGET_FPS)


def _convert(job):
    src, dst, src_fps = job
    dst = Path(dst)
    meta = dst.with_suffix(".json")
    if dst.exists() and meta.exists():
        return json.loads(meta.read_text())["frames"]
    frames = holistic_to_frames(np.load(src, allow_pickle=False), src_fps)
    np.save(dst, frames.astype(np.float16))
    meta.write_text(json.dumps({"fps": TARGET_FPS, "source_fps": src_fps, "width": WIDTH, "height": HEIGHT,
                                "frames": len(frames), "source": "how2sign-holistic"}))
    return len(frames)


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the how2sign_holistic_features folder")
    ap.add_argument("--out", type=Path, required=True, help="dataset root (annotations/ and landmarks/ go here)")
    ap.add_argument("--source-fps", type=float, default=SOURCE_FPS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="per split, for testing")
    args = ap.parse_args()

    (args.out / "annotations").mkdir(parents=True, exist_ok=True)
    report = {}
    for split in args.splits:
        ann = args.src / "metadata" / f"how2sign_realigned_{split}.csv"
        shutil.copy(ann, args.out / "annotations" / ann.name)
        records = read_split(args.out, split)
        if args.limit:
            records = records[: args.limit]
        lm = args.out / "landmarks" / split
        lm.mkdir(parents=True, exist_ok=True)
        jobs, missing = [], 0
        for r in records:
            src = args.src / split / "frontal" / f"{r.sentence_name}_holistic.npy"
            if src.exists():
                jobs.append((str(src), str(lm / f"{r.sentence_name}.npy"), args.source_fps))
            else:
                missing += 1
        frames = []
        with Pool(args.workers) as pool:
            for i, n in enumerate(pool.imap_unordered(_convert, jobs, chunksize=32)):
                frames.append(n)
                if (i + 1) % 2000 == 0:
                    print(f"{split} [{i + 1}/{len(jobs)}]", flush=True)
        report[split] = {"sentences": len(records), "imported": len(jobs), "missing_file": missing,
                         "frames_15fps_median": float(np.median(frames)) if frames else 0}
        print(split, json.dumps(report[split]), flush=True)
    (args.out / "import_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
