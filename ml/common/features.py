"""Landmark features shared by training (Python) and the browser (src/ml/features.ts).

Any change here MUST be mirrored in src/ml/features.ts and FEATURE_VERSION bumped.
`python -m tools.make_parity_fixture` writes a fixture that the TypeScript tests
compare against, so a mismatch fails `npm test`.

Input: packed frames [T, 553, 3] (see layout.py), NaN where a part is missing,
coordinates normalised to the image (0..1) as MediaPipe returns them, plus the
image aspect ratio (width / height).

Per frame we compute (F = 786 values):
  G  global positions  (127 points x 2)  relative to the shoulders, scaled by shoulder width
  H  hand shape        (2 hands x 21 x 3) relative to the wrist, scaled by palm length
  FL face shape        (74 points x 2)   relative to the nose tip, scaled by eye distance
  V  velocity of G     (127 x 2)         G[t] - G[t-1] when the part is present in both
  P  presence flags    (4)               pose, face, left hand, right hand

Missing parts produce zeros plus a 0 presence flag. The shoulder anchor is
carried forward causally (last valid value), so the browser can compute the
same features frame by frame while streaming.
"""
from __future__ import annotations

import numpy as np

from .layout import FACE_START, LEFT_START, POSE_START, RIGHT_START, HAND_POINTS

FEATURE_VERSION = 1
TARGET_FPS = 15

# Upper-body pose: nose, eyes, ears, shoulders, elbows, wrists.
POSE_SEL = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
# Face subset for non-manual markers: lips, eyebrows, eyes, nose, face outline.
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
EYES = [33, 133, 159, 145, 362, 263, 386, 374]
NOSE = [1, 4]
OVAL = [10, 152, 234, 454]
FACE_SEL = LIPS + LEFT_BROW + RIGHT_BROW + EYES + NOSE + OVAL  # 74

N_POSE = len(POSE_SEL)
N_FACE = len(FACE_SEL)
N_GLOBAL = N_POSE + N_FACE + 2 * HAND_POINTS  # 127
FEATURE_DIM = N_GLOBAL * 2 + 2 * HAND_POINTS * 3 + N_FACE * 2 + N_GLOBAL * 2 + 4  # 786

FALLBACK_CENTER_Y = 0.6
FALLBACK_SCALE = 0.35
MIN_SCALE = 1e-3
CLIP = 10.0

# Face indices used as local anchors.
FACE_NOSE_TIP = 1
FACE_EYE_OUTER_R = 33
FACE_EYE_OUTER_L = 263


def _present(frame: np.ndarray, start: int) -> bool:
    return not np.isnan(frame[start, 0])


class FeatureState:
    """Causal state carried between frames (shoulder anchor and previous positions)."""

    def __init__(self, aspect: float):
        self.aspect = float(aspect)
        self.center: tuple[float, float] | None = None
        self.scale: float | None = None
        self.prev_global: np.ndarray | None = None
        self.prev_presence: np.ndarray | None = None


def frame_features(frame: np.ndarray, state: FeatureState) -> np.ndarray:
    """Features for one packed frame [553, 3]. Updates `state`."""
    a = state.aspect
    f = frame.astype(np.float64, copy=True)
    f[:, 0] *= a
    f[:, 2] *= a

    pose_ok = _present(f, POSE_START)
    face_ok = _present(f, FACE_START)
    left_ok = _present(f, LEFT_START)
    right_ok = _present(f, RIGHT_START)

    if pose_ok:
        ls = f[POSE_START + 11]
        rs = f[POSE_START + 12]
        s = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        if s >= MIN_SCALE:
            state.center = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            state.scale = s
    if state.center is None or state.scale is None:
        cx, cy, sc = 0.5 * a, FALLBACK_CENTER_Y, FALLBACK_SCALE
    else:
        (cx, cy), sc = state.center, state.scale

    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    g = np.zeros((N_GLOBAL, 2), dtype=np.float64)
    pres_pts = np.zeros(N_GLOBAL, dtype=bool)

    def put_global(row: int, idx: list[int] | range, start: int, ok: bool):
        if not ok:
            return
        pts = f[[start + i for i in idx], :2]
        g[row:row + len(idx), 0] = (pts[:, 0] - cx) / sc
        g[row:row + len(idx), 1] = (pts[:, 1] - cy) / sc
        pres_pts[row:row + len(idx)] = True

    put_global(0, POSE_SEL, POSE_START, pose_ok)
    put_global(N_POSE, FACE_SEL, FACE_START, face_ok)
    put_global(N_POSE + N_FACE, range(HAND_POINTS), LEFT_START, left_ok)
    put_global(N_POSE + N_FACE + HAND_POINTS, range(HAND_POINTS), RIGHT_START, right_ok)

    o = 0
    out[o:o + N_GLOBAL * 2] = g.reshape(-1)
    o += N_GLOBAL * 2

    for start, ok in ((LEFT_START, left_ok), (RIGHT_START, right_ok)):
        if ok:
            h = f[start:start + HAND_POINTS]
            w = h[0]
            hs = max(float(np.hypot(h[9, 0] - w[0], h[9, 1] - w[1])), MIN_SCALE)
            out[o:o + HAND_POINTS * 3] = ((h - w) / hs).reshape(-1)
        o += HAND_POINTS * 3

    if face_ok:
        pts = f[[FACE_START + i for i in FACE_SEL], :2]
        nose = f[FACE_START + FACE_NOSE_TIP]
        er = f[FACE_START + FACE_EYE_OUTER_R]
        el = f[FACE_START + FACE_EYE_OUTER_L]
        fs = max(float(np.hypot(er[0] - el[0], er[1] - el[1])), MIN_SCALE)
        out[o:o + N_FACE * 2] = ((pts - nose[:2]) / fs).reshape(-1)
    o += N_FACE * 2

    if state.prev_global is not None:
        both = pres_pts & state.prev_presence
        v = np.where(both[:, None], g - state.prev_global, 0.0)
        out[o:o + N_GLOBAL * 2] = v.reshape(-1)
    o += N_GLOBAL * 2

    out[o:o + 4] = [float(pose_ok), float(face_ok), float(left_ok), float(right_ok)]

    state.prev_global = g
    state.prev_presence = pres_pts
    np.clip(out, -CLIP, CLIP, out=out)
    # A part detected with some points missing must not poison the model input.
    np.nan_to_num(out, copy=False, nan=0.0)
    return out.astype(np.float32)


def sequence_features(frames: np.ndarray, aspect: float) -> np.ndarray:
    """Features for a whole sequence [T, 553, 3] -> [T, FEATURE_DIM]."""
    state = FeatureState(aspect)
    if len(frames) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([frame_features(fr, state) for fr in frames])
