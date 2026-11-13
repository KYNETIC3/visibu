"""Check exported ONNX models against the PyTorch checkpoint.

1. Tensor parity: encoder memory / CTC log-probs and decoder log-probs on random
   inputs of several lengths (max abs difference).
2. End-to-end parity: the n-best suggestions from beam search on ONNX Runtime
   (CPU) vs PyTorch on real prepared examples, if a manifest split is given.
Also reports ONNX Runtime CPU latency.

  python -m export.validate_onnx --checkpoint checkpoints/slt-base/best.pt --split val --limit 50
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_DIM, sequence_features  # noqa: E402
from common.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab  # noqa: E402
from models.beam import BeamConfig, beam_search  # noqa: E402
from export.export_onnx import DecoderExport, EncoderExport  # noqa: E402
from training.data import load_manifest  # noqa: E402
from training.evaluate import load_model  # noqa: E402


BEAM = BeamConfig(beam=4, n_best=3, no_repeat_ngram=3, banned=(PAD_ID, BOS_ID, UNK_ID))


def ort_beam(enc, dec, feats: np.ndarray) -> list[list[int]]:
    """What the browser does: encoder once, then beam search over decoder.onnx."""
    memory, _ = enc.run(None, {"features": feats[None].astype(np.float32)})

    def step(prefixes):
        mem = np.repeat(memory, len(prefixes), axis=0)
        return dec.run(None, {"memory": mem, "tokens": np.array(prefixes, dtype=np.int64)})[0]

    return [h.tokens for h in beam_search(step, BOS_ID, EOS_ID, BEAM)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, default=ML_ROOT / "exports" / "visibu-slt")
    ap.add_argument("--prepared", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared")
    ap.add_argument("--split", help="manifest for end-to-end parity, e.g. val or smoke_eval")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    model, _ = load_model(args.checkpoint)
    enc_t, dec_t = EncoderExport(model).eval(), DecoderExport(model).eval()
    so = ort.SessionOptions()
    enc = ort.InferenceSession(str(args.onnx / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(args.onnx / "decoder.onnx"), so, providers=["CPUExecutionProvider"])

    worst = 0.0
    rng = np.random.default_rng(0)
    for T in (8, 33, 120, 301):
        x = rng.normal(size=(1, T, FEATURE_DIM)).astype(np.float32)
        with torch.no_grad():
            m_t, c_t = enc_t(torch.from_numpy(x))
        m_o, c_o = enc.run(None, {"features": x})
        vocab_size = c_o.shape[-1]
        B = T % 4 + 1  # several beams at once, as in beam search
        tok = np.array([[BOS_ID] + list(rng.integers(4, vocab_size, size=T % 7 + 1)) for _ in range(B)], dtype=np.int64)
        m_b = np.repeat(m_o, B, axis=0)
        with torch.no_grad():
            d_t = dec_t(m_t.expand(B, -1, -1), torch.from_numpy(tok))
        d_o = dec.run(None, {"memory": m_b, "tokens": tok})[0]
        diffs = [float(np.abs(m_t.numpy() - m_o).max()), float(np.abs(c_t.numpy() - c_o).max()),
                 float(np.abs(d_t.numpy() - d_o).max())]
        worst = max(worst, *diffs)
        print(f"T={T:4d} S={m_o.shape[1]:4d} B={B}  max|diff| memory {diffs[0]:.2e}  ctc {diffs[1]:.2e}  decoder {diffs[2]:.2e}")
    result = {"tensor_max_abs_diff": worst, "tensor_parity_ok": worst < args.tol}

    if args.split:
        prep = json.loads((args.prepared / "prep_config.json").read_text())
        vocab = Vocab.load(args.onnx / "vocab.json")
        rows = load_manifest(args.prepared / f"{args.split}.jsonl")[: args.limit]
        same, lat = 0, []
        for r in rows:
            feats = sequence_features(np.load(Path(prep["landmarks_root"]) / r["landmarks"]).astype(np.float32), r["aspect"])
            t0 = time.perf_counter()
            o = ort_beam(enc, dec, feats)
            lat.append((time.perf_counter() - t0) * 1000)
            with torch.no_grad():
                mem, _ = model.encoder(torch.from_numpy(feats)[None], None)
            t = [h.tokens for h in model.beam(mem, BOS_ID, EOS_ID, BEAM)]
            same += [vocab.decode(x) for x in o] == [vocab.decode(x) for x in t]
        result.update({"e2e_examples": len(rows), "e2e_identical_suggestions": same,
                       "ort_cpu_latency_ms_median": round(st.median(lat), 1)})
    print(json.dumps(result, indent=2))
    if not result["tensor_parity_ok"] or (args.split and result["e2e_identical_suggestions"] < len(rows)):
        sys.exit(1)


if __name__ == "__main__":
    main()
