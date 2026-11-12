"""Export the isolated-sign classifier (pretrain_signs.py, 250 everyday signs from
Kaggle asl-signs) for the app's word mode.

  signs.onnx   in   features   float32 [1, T, 786]   (the same features as the sentence model)
               out  probs      float32 [1, 250]      softmax over the signs in labels.json order
                    embedding  float32 [1, 256]      the clip's averaged encoder output, used as a
                                                     "fingerprint" to match signs the user teaches

The manifest's "matching" block holds what the app needs to compare
fingerprints: a centring vector (mean fingerprint of TRAIN clips) and a
similarity threshold measured on simulated repeats (see calibrate()).

Writes <out>/signs.onnx, labels.json, manifest.json; --install copies them to
../public/models/visibu-signs/. The manifest records the held-out-participant
test accuracy from the training log, so the app never claims more than that.

  python -m export.export_signs_onnx --checkpoint checkpoints/pretrain-signs/best.pt --install
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_DIM, FEATURE_VERSION, TARGET_FPS  # noqa: E402
from models.sign_encoder import SignEncoder  # noqa: E402
from training.pretrain_signs import SignClassifier  # noqa: E402

APP_DIR = ML_ROOT.parent / "public" / "models" / "visibu-signs"
OPSET = 18


class Export(nn.Module):
    def __init__(self, model: SignClassifier):
        super().__init__()
        self.model = model

    def forward(self, features):
        memory, _ = self.model.encoder(features, None)
        pooled = memory.mean(1)
        return self.model.head(pooled).softmax(-1), pooled


def calibrate(wrapped: "Export", prepared: Path, n: int = 300, takes: int = 3, seed: int = 0) -> dict:
    """Threshold for "this is the same sign" on fingerprints, from simulated repeats.

    A "taught" sign = the average of `takes` augmented copies of one clip (speed,
    rotation, scale, shift, hand occlusion, as in training); a "repeat" = one
    more augmented copy of the same clip; "different" = a copy of another clip.
    The centring vector comes from TRAIN clips; the pairs from VAL clips.
    Real repeats by a person vary more than augmentations, so the measured
    true-match rate is optimistic.
    """
    import random

    from common.features import sequence_features
    from training.data import augment, load_manifest

    cfg = json.loads((prepared / "prep_config.json").read_text())
    root = Path(cfg["landmarks_root"])

    def emb(frames, aspect):
        x = torch.from_numpy(sequence_features(frames, aspect)).unsqueeze(0)
        with torch.no_grad():
            return wrapped(x)[1][0].numpy()

    def short(rows):
        return [r for r in rows if 8 <= r["frames"] <= 60]

    rng = random.Random(seed)
    train = rng.sample(short(load_manifest(prepared / "train.jsonl")), n)
    mean = np.mean([emb(np.load(root / r["landmarks"]).astype(np.float32), r["aspect"]) for r in train], 0)

    def unit(v):
        v = v - mean
        return v / max(np.linalg.norm(v), 1e-8)

    val = rng.sample(short(load_manifest(prepared / "val.jsonl")), n)
    protos, repeats = [], []
    for r in val:
        f = np.load(root / r["landmarks"]).astype(np.float32)
        views = [unit(emb(augment(f, random.Random(rng.random())), r["aspect"])) for _ in range(takes + 1)]
        proto = np.mean(views[:takes], 0)  # as the app does: average the centred unit takes, then renormalise
        protos.append(proto / max(np.linalg.norm(proto), 1e-8))
        repeats.append(views[takes])
    P, R = np.stack(protos), np.stack(repeats)
    sims = P @ R.T
    same = np.diag(sims)
    diff = sims[~np.eye(len(sims), dtype=bool)]
    # Strict: at most 1% of different-sign pairs may count as a match.
    thr = float(np.quantile(diff, 0.99))
    return {
        "mean": [round(float(x), 6) for x in mean],
        "threshold": round(thr, 4),
        "simulated": {
            "pairs_same": int(len(same)), "pairs_different": int(len(diff)), "takes": takes,
            "true_match_rate": round(float((same >= thr).mean()), 4),
            "false_match_rate": round(float((diff >= thr).mean()), 4),
            "same_median": round(float(np.median(same)), 4), "different_median": round(float(np.median(diff)), 4),
            "note": "How2Sign val clips with training augmentations; real repeats vary more, so true_match_rate is optimistic.",
        },
    }


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ML_ROOT / "exports" / "visibu-signs")
    ap.add_argument("--version", default=time.strftime("%Y%m%d"))
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--calibrate", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared",
                    help="prepared How2Sign folder (train.jsonl, val.jsonl) for the matching threshold")
    args = ap.parse_args()

    run = args.checkpoint.parent
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck["meta"]["feature_version"] != FEATURE_VERSION:
        sys.exit("Checkpoint uses a different feature version.")
    labels_by_index = sorted(json.loads((run / "labels.json").read_text()).items(), key=lambda kv: kv[1])
    labels = [k for k, _ in labels_by_index]
    e = ck["encoder_config"]
    model = SignClassifier(SignEncoder(FEATURE_DIM, e["d_model"], e["enc_layers"], e["heads"], e["ff"], 0.0), e["d_model"], len(labels), 0.0)
    model.load_state_dict(ck["model"])
    wrapped = Export(model).eval()

    # Parity with the training-time forward (masked mean pool over a single unpadded clip).
    x = torch.randn(1, 37, FEATURE_DIM)
    with torch.no_grad():
        ref = model(x, torch.tensor([37])).softmax(-1)
        got = wrapped(x)[0]
    assert torch.allclose(ref, got, atol=1e-5), float((ref - got).abs().max())

    args.out.mkdir(parents=True, exist_ok=True)
    T = torch.export.Dim("T", min=2, max=4000)
    torch.onnx.export(wrapped, (x,), args.out / "signs.onnx", input_names=["features"], output_names=["probs", "embedding"],
                      dynamic_shapes={"features": {1: T}}, opset_version=OPSET, dynamo=True, external_data=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out / "signs.onnx"), providers=["CPUExecutionProvider"])
    diff = 0.0
    for t in (2, 15, 90):
        xi = torch.randn(1, t, FEATURE_DIM)
        with torch.no_grad():
            r = [t.numpy() for t in wrapped(xi)]
        o = sess.run(None, {"features": xi.numpy()})
        diff = max(diff, float(np.abs(o[0] - r[0]).max()), float(np.abs(o[1] - r[1]).max()))
    print("onnx vs torch max |diff|", diff)
    assert diff < 1e-4

    (args.out / "labels.json").write_text(json.dumps(labels))
    matching = calibrate(wrapped, args.calibrate)
    print("matching", json.dumps({k: v for k, v in matching.items() if k != "mean"}))
    test = None
    log = run / "train_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "test":
                test = {k: r[k] for k in ("top1", "top5", "n", "participants")}
    manifest = {
        "name": "visibu-signs",
        "version": args.version,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "fps": TARGET_FPS,
        "signs": len(labels),
        "test": test,
        "best_val_top1": ck.get("best"),
        "embedding_dim": e["d_model"],
        "matching": matching,
        "dataset": "Google Isolated Sign Language Recognition (Kaggle asl-signs, PopSign ASL)",
        "licence": "CC BY 4.0 data (Kaggle competition rules 7.A). Attribution: Google, PopSign.",
        "files": {f: {"sha256": sha256(args.out / f), "bytes": (args.out / f).stat().st_size} for f in ("signs.onnx", "labels.json")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("version", "signs", "test", "files")}, indent=2))
    if args.install:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        for f in ("signs.onnx", "labels.json", "manifest.json"):
            shutil.copy(args.out / f, APP_DIR / f)
        print(f"installed to {APP_DIR}")


if __name__ == "__main__":
    main()
