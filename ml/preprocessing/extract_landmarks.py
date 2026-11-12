"""Extract MediaPipe Holistic landmarks from How2Sign sentence clips.

Uses the SAME model file and MediaPipe version as the browser
(public/mediapipe/holistic_landmarker.task, mediapipe 1.2.2), and writes frames
in the browser's packed layout, so training and inference see the same input.

Frames are sampled at TARGET_FPS (15) by nearest frame, matching the browser's
resampling in src/ml/sequence.ts. Only landmark numbers are stored (float16),
never video.

Output per clip: <out>/<split>/<SENTENCE_NAME>.npy  [T, 553, 3] float16
plus <out>/<split>/<SENTENCE_NAME>.json {fps, width, height, frames, source_frames}.
Resumable: existing outputs are skipped.

  python -m preprocessing.extract_landmarks --split val --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import TARGET_FPS  # noqa: E402
from common.layout import (DIMS, FACE_POINTS, FACE_START, HAND_POINTS, LEFT_START, POSE_POINTS,  # noqa: E402
                           POSE_START, RIGHT_START, TOTAL_POINTS)

DEFAULT_MODEL = ML_ROOT.parent / "public" / "mediapipe" / "holistic_landmarker.task"
DEFAULT_ROOT = ML_ROOT / "datasets" / "how2sign"

_landmarker = None
_next_ts = 0
# Gap between clips on one landmarker's clock. Measured on How2Sign val: reusing
# one landmarker with this gap gives the same landmarks as a fresh one (mean
# difference < 0.001 of image width) and saves ~1.2 s of setup per clip.
CLIP_GAP_MS = 100_000


def _init_worker(model_path: str) -> None:
    global _landmarker
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
    _landmarker = HolisticLandmarker.create_from_options(HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
    ))


def _points(landmarks) -> list | None:
    if landmarks is None:
        return None
    # The Python result holds one person: a flat list of landmarks.
    if len(landmarks) and isinstance(landmarks[0], (list, tuple)):
        landmarks = landmarks[0]
    return landmarks if len(landmarks) else None


def pack(result) -> np.ndarray:
    out = np.full((TOTAL_POINTS, DIMS), np.nan, dtype=np.float32)
    for start, count, lms in (
        (POSE_START, POSE_POINTS, _points(result.pose_landmarks)),
        (LEFT_START, HAND_POINTS, _points(result.left_hand_landmarks)),
        (RIGHT_START, HAND_POINTS, _points(result.right_hand_landmarks)),
        (FACE_START, FACE_POINTS, _points(result.face_landmarks)),
    ):
        if lms is None:
            continue
        for i, p in enumerate(lms[:count]):
            out[start + i] = (p.x, p.y, p.z)
    return out


def extract_clip(args: tuple[str, str, str, int]) -> tuple[str, str]:
    """Returns (name, status). Runs in a worker process."""
    import cv2
    import mediapipe as mp

    global _landmarker, _next_ts
    video_path, out_npy, model_path, max_width = args
    name = Path(video_path).stem
    if os.path.exists(out_npy):
        return name, "skipped"
    if _landmarker is None:
        _init_worker(model_path)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return name, "unreadable"
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or n_src <= 0 or width <= 0 or height <= 0:
        cap.release()
        return name, "bad-metadata"

    duration = n_src / fps
    n_out = max(1, int(math.floor(duration * TARGET_FPS)))
    wanted = {}
    for k in range(n_out):
        wanted.setdefault(min(n_src - 1, int(round(k * fps / TARGET_FPS))), []).append(k)

    frames = np.full((n_out, TOTAL_POINTS, DIMS), np.nan, dtype=np.float32)
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in wanted:
            if max_width and bgr.shape[1] > max_width:
                # Half resolution: 28% faster, same detections (see README).
                f = max_width / bgr.shape[1]
                bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = _next_ts + int(round(wanted[idx][0] * 1000 / TARGET_FPS))
            packed = pack(_landmarker.detect_for_video(image, ts_ms))
            for k in wanted[idx]:
                frames[k] = packed
        idx += 1
    cap.release()
    _next_ts += int(round(n_out * 1000 / TARGET_FPS)) + CLIP_GAP_MS

    tmp = out_npy + ".tmp.npy"
    np.save(tmp, frames.astype(np.float16))
    os.replace(tmp, out_npy)
    Path(out_npy[:-4] + ".json").write_text(json.dumps({
        "fps": TARGET_FPS, "source_fps": fps, "width": width, "height": height,
        "frames": n_out, "source_frames": n_src, "decoded_frames": idx, "detect_max_width": max_width,
    }))
    return name, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--clips", type=Path, help="folder of <SENTENCE_NAME>.mp4 (default: <root>/clips/<split>)")
    ap.add_argument("--out", type=Path, help="default: <root>/landmarks")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=2,
                    help="MediaPipe already uses several threads; more than 2-3 processes did not help on an M2")
    ap.add_argument("--max-width", type=int, default=640, help="downscale frames wider than this before detection (0 = off)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (for smoke tests)")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"Missing {args.model}. Run `npm run assets` in the repo root first.")
    clips = args.clips or args.root / "clips" / args.split
    out = (args.out or args.root / "landmarks") / args.split
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(clips.rglob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 files under {clips}")
    if args.limit:
        videos = videos[: args.limit]
    jobs = [(str(v), str(out / f"{v.stem}.npy"), str(args.model), args.max_width) for v in videos]

    started = time.time()
    counts: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for i, (name, status) in enumerate(pool.imap_unordered(extract_clip, jobs, chunksize=8), 1):
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "skipped"):
                print(f"[warn] {name}: {status}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                rate = i / max(time.time() - started, 1e-6)
                print(f"[{i}/{len(jobs)}] {counts}  {rate:.2f} clips/s", flush=True)
    (out / "_extract_log.json").write_text(json.dumps({
        "split": args.split, "clips": len(jobs), "counts": counts, "seconds": round(time.time() - started, 1),
        "model": args.model.name, "target_fps": TARGET_FPS, "max_width": args.max_width,
    }, indent=2))


if __name__ == "__main__":
    main()
