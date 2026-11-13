"""Export a checkpoint to two ONNX models for ONNX Runtime Web, plus a manifest.

  recognition encoder  encoder.onnx
      in   features      float32 [1, T, 786]   (T >= 1 frames at 15 fps, src/ml/features.ts)
      out  memory        float32 [1, S, D]     S = ceil(T / 2)
           ctc_log_probs float32 [1, S, V]
  translation decoder  decoder.onnx
      in   memory        float32 [B, S, D]     B = live beams; the encoder memory repeated B times
           tokens        int64   [B, L]        each row starts with <bos>
      out  next_log_probs float32 [B, V]       for the token after the last one

Writes to <out>/: encoder.onnx, decoder.onnx, vocab.json, manifest.json.
With --install, also copies them to ../public/models/visibu-slt/ for the app.

  python -m export.export_onnx --checkpoint checkpoints/slt-base/best.pt --stage research --install
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.features import FEATURE_DIM, FEATURE_VERSION, TARGET_FPS  # noqa: E402
from common.tokenizer import BOS_ID, CTC_BLANK_ID, EOS_ID, PAD_ID, UNK_ID  # noqa: E402
from training.evaluate import load_model  # noqa: E402

APP_MODEL_DIR = ML_ROOT.parent / "public" / "models" / "visibu-slt"
OPSET = 18


class EncoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.ctc = model.encoder, model.ctc

    def forward(self, features):
        memory, _ = self.encoder(features, None)
        return memory, self.ctc(memory)


class DecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, memory, tokens):
        return self.decoder(tokens, memory, None)[:, -1].log_softmax(-1)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export(model, out: Path) -> None:
    enc, dec = EncoderExport(model).eval(), DecoderExport(model).eval()
    feats = torch.randn(1, 40, FEATURE_DIM)
    with torch.no_grad():
        memory, _ = enc(feats)
    memory = memory.expand(2, -1, -1).contiguous()
    tokens = torch.tensor([[BOS_ID, 5, 6], [BOS_ID, 7, 8]], dtype=torch.long)
    T, S, L = torch.export.Dim("T", min=2, max=4000), torch.export.Dim("S", min=1, max=2000), torch.export.Dim("L", min=1, max=256)
    B = torch.export.Dim("B", min=1, max=16)
    torch.onnx.export(enc, (feats,), out / "encoder.onnx", input_names=["features"],
                      output_names=["memory", "ctc_log_probs"], dynamic_shapes={"features": {1: T}},
                      opset_version=OPSET, dynamo=True, external_data=False)
    torch.onnx.export(dec, (memory, tokens), out / "decoder.onnx", input_names=["memory", "tokens"],
                      output_names=["next_log_probs"], dynamic_shapes={"memory": {0: B, 1: S}, "tokens": {0: B, 1: L}},
                      opset_version=OPSET, dynamo=True, external_data=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared")
    ap.add_argument("--out", type=Path, default=ML_ROOT / "exports" / "visibu-slt")
    ap.add_argument("--stage", choices=["smoke", "research"], required=True,
                    help="smoke = pipeline test only, output is meaningless; research = trained and evaluated")
    ap.add_argument("--eval-report", type=Path, help="evaluation JSON to summarise in the manifest")
    ap.add_argument("--version", default=time.strftime("%Y%m%d"))
    ap.add_argument("--install", action="store_true")
    args = ap.parse_args()

    model, ck = load_model(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    export(model, args.out)
    shutil.copy(args.prepared / "vocab.json", args.out / "vocab.json")

    evaluation = None
    if args.eval_report:
        rep = json.loads(args.eval_report.read_text())
        evaluation = {"split": rep["split"], "examples": rep["examples"], "translation": rep["translation"],
                      "ctc_head": rep["ctc_head"]}
    manifest = {
        "name": "visibu-slt",
        "version": args.version,
        "stage": args.stage,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "fps": TARGET_FPS,
        "min_frames": 8,
        "max_frames": 450,
        "max_tokens": 60,
        "decoding": {"beam": 4, "n_best": 3, "no_repeat_ngram": 3},
        "special_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID, "ctc_blank": CTC_BLANK_ID},
        "ctc_targets": "gloss" if ck["meta"].get("ctc_source") == "gloss" else "english-words",
        "model": ck["config"],
        "training": {k: ck["meta"][k] for k in ("train_manifest", "val_manifest", "train_examples", "val_examples")},
        "evaluation": evaluation,
        "dataset": "How2Sign (https://how2sign.github.io), re-aligned English translations, front-view clips",
        "licence": "Weights are derived from How2Sign, licensed CC BY-NC 4.0. Non-commercial research use only.",
        "files": {},
    }
    for f in ("encoder.onnx", "decoder.onnx", "vocab.json"):
        manifest["files"][f] = {"sha256": sha256(args.out / f), "bytes": (args.out / f).stat().st_size}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("version", "stage", "files")}, indent=2))

    if args.install:
        APP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for f in ("encoder.onnx", "decoder.onnx", "vocab.json", "manifest.json"):
            shutil.copy(args.out / f, APP_MODEL_DIR / f)
        print(f"installed to {APP_MODEL_DIR}")


if __name__ == "__main__":
    main()
