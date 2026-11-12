"""Import Google's "Isolated Sign Language Recognition" Kaggle data (asl-signs)
into Visibu's packed landmark format, for pre-training the sign encoder.

The competition data is MediaPipe Holistic landmarks (no video): one parquet
file per clip with columns frame, row_id, type, landmark_index, x, y, z, where
type is face (468 points), left_hand (21), pose (33) or right_hand (21). That
is the same landmarker family the app uses, so no extraction is needed.

Differences from the app's frames, and how they are handled:
  * face: 468 points, not 478 (no iris). The features use face points <= 454,
    so the 10 iris slots are left NaN.
  * frame rate: not published. --source-fps (default 30) is used to resample
    to the training rate (15 fps) by nearest frame, as for How2Sign.
  * aspect ratio: not published. --aspect sets it (see README).

Reads the parquet files straight out of the downloaded zip (no 57 GB
extraction) and writes, per clip, float16 [T, 553, 3] .npy files plus
manifests split BY PARTICIPANT, so no signer appears in two splits.

  python -m preprocessing.import_asl_signs --zip datasets/asl-signs/raw/asl-signs.zip
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import TARGET_FPS  # noqa: E402
from common.layout import FACE_START, LEFT_START, POSE_START, RIGHT_START, TOTAL_POINTS  # noqa: E402

ROOT = ML_ROOT / "datasets" / "asl-signs"
OFFSETS = {"pose": (POSE_START, 33), "left_hand": (LEFT_START, 21), "right_hand": (RIGHT_START, 21),
           "face": (FACE_START, 468)}
# Two held-out participants for validation and two for a final test. Chosen by
# id order, not by results. Never tune on test.
VAL_PARTICIPANTS = {"2044", "4718"}
TEST_PARTICIPANTS = {"16069", "18796"}


def parquet_to_frames(data: bytes) -> np.ndarray:
    """Parquet bytes -> [F, 553, 3] float32, NaN where a part was not detected."""
    import pyarrow.parquet as pq

    t = pq.read_table(io.BytesIO(data), columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frame = t.column("frame").to_numpy()
    types = t.column("type").to_pylist()
    lmi = t.column("landmark_index").to_numpy().astype(np.int64)
    xyz = np.stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in "xyz"], 1)
    uniq = np.unique(frame)
    pos = np.searchsorted(uniq, frame)
    out = np.full((len(uniq), TOTAL_POINTS, 3), np.nan, dtype=np.float32)
    types = np.asarray(types)
    for name, (start, n) in OFFSETS.items():
        m = (types == name) & (lmi < n)
        out[pos[m], start + lmi[m]] = xyz[m]
    return out


def resample(frames: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS) -> np.ndarray:
    """Nearest-frame resampling, as in extract_landmarks.py."""
    n = max(1, int(round(len(frames) * dst_fps / src_fps)))
    idx = np.clip(np.round(np.arange(n) * src_fps / dst_fps).astype(int), 0, len(frames) - 1)
    return frames[idx]


def split_of(participant: str) -> str:
    if participant in TEST_PARTICIPANTS:
        return "test"
    if participant in VAL_PARTICIPANTS:
        return "val"
    return "train"


# Cheek-to-cheek / forehead-to-chin ratio of a face, measured on How2Sign where
# the 16:9 aspect is known (300 clips, median 0.778). Used to check --aspect.
TRUE_FACE_RATIO = 0.778


def face_ratio(frames: np.ndarray) -> float:
    face = frames[:, FACE_START:FACE_START + 468]
    ok = ~np.isnan(face[:, 0, 0])
    if not ok.any():
        return float("nan")
    f = face[ok]
    w = np.linalg.norm(f[:, 234, :2] - f[:, 454, :2], axis=1)
    h = np.linalg.norm(f[:, 10, :2] - f[:, 152, :2], axis=1)
    return float(np.median(w / np.maximum(h, 1e-6)))


class Reader:
    """Reads clip files from the Kaggle zip or from the unpacked folder (Kaggle notebooks)."""

    def __init__(self, zip_path: Path | None, src_dir: Path | None):
        self.zip_path, self.src_dir, self.zf = zip_path, src_dir, None

    def read(self, rel: str) -> bytes:
        if self.src_dir:
            return (self.src_dir / rel).read_bytes()
        if self.zf is None:  # one handle per process
            self.zf = zipfile.ZipFile(self.zip_path)
        return self.zf.read(rel)


_reader: Reader | None = None


def _init(zip_path, src_dir):
    global _reader
    _reader = Reader(zip_path, src_dir)


def _convert(job):
    rel, target, src_fps = job
    target = Path(target)
    if target.exists():
        frames = np.load(target)
    else:
        frames = resample(parquet_to_frames(_reader.read(rel)), src_fps).astype(np.float16)
        np.save(target, frames)
    f = frames.astype(np.float32)
    hands = bool((~np.isnan(f[:, LEFT_START, 0]) | ~np.isnan(f[:, RIGHT_START, 0])).any())
    return len(f), hands, face_ratio(f)


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=ROOT / "raw" / "asl-signs.zip")
    ap.add_argument("--src-dir", type=Path, help="unpacked competition folder instead of the zip (e.g. /kaggle/input/asl-signs)")
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--source-fps", type=float, default=30.0)
    ap.add_argument("--aspect", type=float, default=3 / 4,
                    help="width / height of the source video; 3:4 portrait, estimated from face proportions (README)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lm_dir = args.out / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)
    reader = Reader(None if args.src_dir else args.zip, args.src_dir)
    rows = list(csv.DictReader(io.StringIO(reader.read("train.csv").decode("utf-8"))))
    signs = json.loads(reader.read("sign_to_prediction_index_map.json"))
    if args.limit:
        rows = rows[: args.limit]
    jobs = [(r["path"], str(lm_dir / f"{r['sequence_id']}.npy"), args.source_fps) for r in rows]

    manifests = {"train": [], "val": [], "test": []}
    empty, ratios = 0, []
    with Pool(args.workers, initializer=_init, initargs=(reader.zip_path, args.src_dir)) as pool:
        for i, (r, (n, hands, ratio)) in enumerate(zip(rows, pool.imap(_convert, jobs, chunksize=64))):
            if ratio == ratio:
                ratios.append(ratio)
            if not hands:
                empty += 1
                continue
            manifests[split_of(r["participant_id"])].append({
                "name": r["sequence_id"], "landmarks": f"{r['sequence_id']}.npy", "frames": n, "aspect": args.aspect,
                "sign": r["sign"], "label": signs[r["sign"]], "signer": r["participant_id"],
            })
            if (i + 1) % 5000 == 0:
                print(f"[{i + 1}/{len(rows)}]", flush=True)

    for split, items in manifests.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in items)
    med = float(np.median(ratios)) if ratios else float("nan")
    report = {
        "clips": {s: len(v) for s, v in manifests.items()}, "skipped_no_hands": empty,
        "signs": len(signs), "participants": {s: sorted({x["signer"] for x in v}, key=int) for s, v in manifests.items()},
        "frames_15fps": {"median": float(np.median([x["frames"] for v in manifests.values() for x in v] or [0]))},
        "assumed": {"source_fps": args.source_fps, "aspect": args.aspect},
        "aspect_check": {"face_ratio_median": round(med, 4), "implied_aspect": round(TRUE_FACE_RATIO / med, 3) if ratios else None,
                         "implied_aspect_p10_p90": [round(TRUE_FACE_RATIO / x, 3) for x in np.percentile(ratios, [90, 10])] if ratios else None,
                         "clips": len(ratios)},
        "per_sign_min_train": min(Counter(x["sign"] for x in manifests["train"]).values(), default=0),
    }
    (args.out / "import_report.json").write_text(json.dumps(report, indent=2))
    (args.out / "labels.json").write_text(json.dumps(signs, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
