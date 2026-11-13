"""Write src/ml/__fixtures__/parity.json so the TypeScript port of the feature
extraction, tokeniser and CTC decoding is tested against this Python code.

Uses synthetic frames only (random landmarks with dropped parts). No How2Sign
data goes into the fixture, because it is committed to the repository.

  python -m tools.make_parity_fixture
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_DIM, FEATURE_VERSION, sequence_features  # noqa: E402
from common.layout import FACE_START, HAND_POINTS, LEFT_START, POSE_START, RIGHT_START, TOTAL_POINTS  # noqa: E402
from common.tokenizer import detokenize, tokenize  # noqa: E402
from models.beam import BeamConfig, beam_search  # noqa: E402
from models.ctc_head import ctc_greedy_decode  # noqa: E402

OUT = ML_ROOT.parent / "src" / "ml" / "__fixtures__" / "parity.json"


def synthetic_sequence(rng: np.random.Generator, T: int) -> np.ndarray:
    base = rng.uniform(0.2, 0.8, size=(TOTAL_POINTS, 3)).astype(np.float32)
    base[:, 2] = rng.normal(0, 0.05, size=TOTAL_POINTS)
    frames = []
    for t in range(T):
        f = base + rng.normal(0, 0.01, size=base.shape).astype(np.float32) * (t + 1) ** 0.5
        f = f.astype(np.float32)
        if t % 5 == 1:
            f[LEFT_START:LEFT_START + HAND_POINTS] = np.nan
        if t % 7 == 3:
            f[RIGHT_START:RIGHT_START + HAND_POINTS] = np.nan
        if t in (0, 2):
            f[POSE_START:POSE_START + 33] = np.nan  # no anchor yet at t=0, lost again at t=2
        if t == 4:
            f[FACE_START:] = np.nan
        frames.append(f)
    return np.stack(frames)


def beam_cases(rng: np.random.Generator) -> list[dict]:
    """A toy bigram 'decoder' (log-probs depend on the last token only), so the
    TypeScript beam search can be checked against this one without ONNX."""
    V, bos, eos = 12, 1, 2
    cases = []
    for beam, n_best, nrep, max_len, peaky in ((4, 3, 3, 12, 2.0), (3, 3, 2, 8, 1.0), (1, 1, 0, 10, 3.0), (5, 3, 3, 6, 0.5)):
        logits = rng.normal(0, peaky, size=(V, V))
        logits[:, 4] += 1.5  # a favourite token, so repetition blocking matters
        table = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        cfg = BeamConfig(beam=beam, n_best=n_best, max_len=max_len, no_repeat_ngram=nrep, banned=(0, bos, 3))
        hyps = beam_search(lambda prefixes: table[[p[-1] for p in prefixes]], bos, eos, cfg)
        cases.append({"table": table.tolist(), "bos": bos, "eos": eos,
                      "config": {"beam": beam, "nBest": n_best, "maxLen": max_len, "noRepeatNgram": nrep,
                                 "banned": [0, bos, 3]},
                      "expected": [{"tokens": h.tokens, "score": h.score, "finished": h.finished} for h in hyps]})
    return cases


def main() -> None:
    rng = np.random.default_rng(42)
    seqs = []
    for T, aspect in ((9, 16 / 9), (5, 4 / 3)):
        frames = synthetic_sequence(rng, T)
        feats = sequence_features(frames, aspect)
        # float32 round trip, as the browser receives Float32Array frames.
        seqs.append({"aspect": aspect, "frames": np.where(np.isnan(frames), None, frames).tolist(),
                     "features": feats.tolist()})
    sentences = [["hi", ",", "i", "want", "a", "large", "coffee", "."], ["i'm", "fine", "!", "thanks"], ["."]]
    texts = ["Hi, I'd like an iced latte — with oat milk, please!", "“Two” SMALL teas?"]
    lp = np.log(np.full((1, 7, 5), 0.01))
    for t, k in enumerate([0, 3, 3, 0, 3, 4, 4]):
        lp[0, t, k] = 0.0
    fixture = {
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "sequences": seqs,
        "detokenize": [{"tokens": s, "text": detokenize(s)} for s in sentences],
        "tokenize": [{"text": t, "tokens": tokenize(t)} for t in texts],
        "beam": beam_cases(rng),
        "ctc": {"log_probs": lp[0].tolist(), "decoded": ctc_greedy_decode(__import__("torch").tensor(lp))[0]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
