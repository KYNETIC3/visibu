"""Pre-train the sign encoder on isolated signs (Kaggle asl-signs, 250 signs).

The encoder (the same SignEncoder the app runs) reads a clip, its outputs are
averaged over time, and a linear layer predicts which of the 250 signs it is.
Afterwards, train_slt.py --init-encoder <this run>/best.pt starts sentence
translation from these weights instead of from random ones. The classifier
head is thrown away; the app never shows these 250 labels.

Splits are by participant (see import_asl_signs.py). Validation accuracy is
reported on held-out participants; the test participants are only touched by
--evaluate-test, once, at the end.

  python -m training.pretrain_signs --out checkpoints/pretrain-signs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.device import pick_device, seed_everything  # noqa: E402
from common.features import FEATURE_DIM, FEATURE_VERSION, sequence_features  # noqa: E402
from models.sign_encoder import SignEncoder  # noqa: E402
from training.data import BucketSampler, augment, load_manifest  # noqa: E402

DATA = ML_ROOT / "datasets" / "asl-signs"


class SignClipDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, train: bool, seed: int = 0):
        self.rows, self.root, self.train, self.seed, self.epoch = rows, root, train, seed, 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        frames = np.load(self.root / r["landmarks"], mmap_mode="r").astype(np.float32)
        if self.train:
            rng = random.Random(hash((self.seed, self.epoch, i)))
            frames = augment(frames, rng)
        return torch.from_numpy(sequence_features(frames, r["aspect"])), r["label"]


def collate(batch):
    T = max(len(x) for x, _ in batch)
    feats = torch.zeros(len(batch), T, FEATURE_DIM)
    for j, (x, _) in enumerate(batch):
        feats[j, : len(x)] = x
    return feats, torch.tensor([len(x) for x, _ in batch]), torch.tensor([y for _, y in batch])


class SignClassifier(nn.Module):
    def __init__(self, encoder: SignEncoder, d_model: int, n_classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, feats, lens):
        memory, pad = self.encoder(feats, lens)
        keep = (~pad).unsqueeze(-1).float() if pad is not None else torch.ones_like(memory[..., :1])
        pooled = (memory * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.head(pooled)


def feature_stats(rows, root, n=3000, seed=0):
    rng = random.Random(seed)
    pick = rng.sample(rows, min(n, len(rows)))
    feats = np.concatenate([sequence_features(np.load(root / r["landmarks"]).astype(np.float32), r["aspect"]) for r in pick])
    return feats.mean(0).tolist(), np.maximum(feats.std(0), 1e-3).tolist()


def evaluate(model, loader, device):
    model.eval()
    correct = top5 = n = 0
    loss = 0.0
    with torch.no_grad():
        for feats, lens, y in loader:
            logits = model(feats.to(device), lens.to(device)).float().cpu()
            loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(-1) == y).sum().item()
            top5 += (logits.topk(5, -1).indices == y[:, None]).any(-1).sum().item()
            n += len(y)
    return {"loss": loss / max(n, 1), "top1": correct / max(n, 1), "top5": top5 / max(n, 1), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    # Must match the sentence model that will start from this encoder (train_slt.py defaults).
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hours", type=float, default=0, help="stop before the next epoch would pass this many hours (0 = no limit)")
    ap.add_argument("--evaluate-test", action="store_true", help="report held-out-participant test accuracy of best.pt, once")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.data / "landmarks"
    labels = json.loads((args.data / "labels.json").read_text())

    def log(obj):
        obj = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **obj}
        print(json.dumps(obj), flush=True)
        with open(args.out / "train_log.jsonl", "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    train_rows = load_manifest(args.data / "train.jsonl")
    val_rows = load_manifest(args.data / "val.jsonl")
    if {r["signer"] for r in train_rows} & {r["signer"] for r in val_rows}:
        sys.exit("Train and validation share participants.")
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]

    def make_model(stats=None):
        enc = SignEncoder(FEATURE_DIM, args.d_model, args.enc_layers, args.heads, args.ff, args.dropout)
        if stats:
            enc.set_stats(*stats)
        return SignClassifier(enc, args.d_model, len(labels), args.dropout)

    if args.evaluate_test:
        ck = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
        model = make_model()
        model.load_state_dict(ck["model"])
        model.to(device)
        test_rows = load_manifest(args.data / "test.jsonl")
        dl = DataLoader(SignClipDataset(test_rows, root, False), batch_sampler=BucketSampler([r["frames"] for r in test_rows], args.batch_size, False),
                        collate_fn=collate, num_workers=args.workers)
        res = evaluate(model, dl, device)
        log({"event": "test", **res, "participants": sorted({r["signer"] for r in test_rows}, key=int)})
        return

    stats = feature_stats(train_rows, root)
    model = make_model(stats).to(device)
    train_ds = SignClipDataset(train_rows, root, True, args.seed)
    sampler = BucketSampler([r["frames"] for r in train_rows], args.batch_size, True, args.seed)
    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, **kw)
    val_dl = DataLoader(SignClipDataset(val_rows, root, False), batch_sampler=BucketSampler([r["frames"] for r in val_rows], args.batch_size, False), **kw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total = args.epochs * len(sampler)
    warm = max(1, min(1000, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start, best, bad, step = 0, -1.0, 0, 0
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad, step = ck["epoch"] + 1, ck["best"], ck["bad"], ck["step"]
    encoder_cfg = {"in_dim": FEATURE_DIM, "d_model": args.d_model, "enc_layers": args.enc_layers, "heads": args.heads,
                   "ff": args.ff, "subsample": 2}
    meta = {"feature_version": FEATURE_VERSION, "encoder": encoder_cfg, "train": len(train_rows), "val": len(val_rows),
            "train_participants": sorted({r["signer"] for r in train_rows}, key=int),
            "val_participants": sorted({r["signer"] for r in val_rows}, key=int),
            "dataset": "Kaggle asl-signs (Google Isolated Sign Language Recognition, PopSign)",
            "args": {k: str(v) for k, v in vars(args).items()}}
    (args.out / "run_config.json").write_text(json.dumps(meta, indent=2))
    log({"event": "start", "device": str(device), "params": sum(p.numel() for p in model.parameters()), **{k: meta[k] for k in ("train", "val")}})

    run_start = time.time()
    for epoch in range(start, args.epochs):
        if args.hours and epoch > start:
            per_epoch = (time.time() - run_start) / (epoch - start)
            if time.time() - run_start + per_epoch > args.hours * 3600:
                log({"event": "time_limit", "epoch": epoch, "hours": args.hours})
                break
        model.train()
        sampler.epoch = train_ds.epoch = epoch
        t0, tot, n = time.time(), 0.0, 0
        for feats, lens, y in train_dl:
            logits = model(feats.to(device), lens.to(device))
            loss = ce(logits, y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item() * len(y); n += len(y)
            if step % 100 == 0:
                log({"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)})
        va = evaluate(model, val_dl, device)
        improved = va["top1"] > best
        best, bad = (va["top1"], 0) if improved else (best, bad + 1)
        log({"epoch": epoch, "train_loss": tot / max(n, 1), "val": va, "best_val_top1": best, "sec": round(time.time() - t0)})
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
              "best": best, "bad": bad, "step": step, "encoder_config": encoder_cfg, "meta": meta}
        torch.save(ck, args.out / "last.pt")
        if improved:
            torch.save(ck, args.out / "best.pt")
        if bad >= args.patience:
            log({"event": "early_stop", "epoch": epoch})
            break
    log({"event": "done", "best_val_top1": best})


if __name__ == "__main__":
    main()
