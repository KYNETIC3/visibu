"""Train the sign-language translation model (encoder + CTC head + decoder) jointly.

  python -m training.train_slt --out checkpoints/slt-base
  python -m training.train_slt --out checkpoints/slt-base --resume      # continue
  python -m training.train_slt --train-manifest smoke_train --val-manifest smoke_eval --out checkpoints/smoke --epochs 3

Prefers Apple MPS, then CUDA (cloud training only), then CPU. On MPS the CTC loss is
computed on CPU because PyTorch has no MPS kernel for it; gradients still flow back.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.device import pick_device, seed_everything  # noqa: E402
from common.features import FEATURE_DIM, FEATURE_VERSION, TARGET_FPS  # noqa: E402
from common.tokenizer import BOS_ID, CTC_BLANK_ID, EOS_ID, PAD_ID, Vocab  # noqa: E402
from models.slt import ModelConfig, SLTModel  # noqa: E402
from training.data import BucketSampler, SignTextDataset, collate, load_manifest  # noqa: E402

DEFAULT_PREPARED = ML_ROOT / "datasets" / "how2sign" / "prepared"


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--train-manifest", default="train")
    ap.add_argument("--val-manifest", default="val")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-model", type=Path,
                    help="start from a whole trained sentence model (e.g. trained/slt-how2sign/model.pt) to fine-tune it; "
                         "--prepared must use that model's vocab.json")
    ap.add_argument("--init-encoder", type=Path,
                    help="start the sign encoder from a pretrain_signs.py checkpoint (its feature statistics too)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=300, help="training clips longer than this are skipped (20 s at 15 fps)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ctc-weight", type=float, default=0.3)
    ap.add_argument("--ctc-source", choices=["auto", "gloss", "words"], default="auto")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12, help="early stopping: epochs without val-loss improvement")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0, help="stop before the next epoch would pass this many hours (0 = no limit)")
    return ap.parse_args(argv)


def run_epoch(model, loader, device, opt=None, sched=None, args=None, log=None, step0=0):
    train = opt is not None
    model.train(train)
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing if train else 0.0)
    ctc = torch.nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)
    tot = {"loss": 0.0, "ce": 0.0, "ctc": 0.0, "tokens": 0, "batches": 0}
    step = step0
    t0 = time.time()
    for b in loader:
        feats = b["feats"].to(device)
        lens = b["feat_lens"].to(device)
        tin, tout = b["tokens_in"].to(device), b["tokens_out"].to(device)
        with torch.set_grad_enabled(train):
            ctc_lp, ctc_lens, logits = model(feats, lens, tin)
            loss_ce = ce(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
            loss_ctc = torch.zeros((), device=device)
            if args.ctc_weight > 0 and b["ctc"].numel() > 0:
                if device.type == "mps":
                    loss_ctc = ctc(ctc_lp.transpose(0, 1).float().cpu(), b["ctc"], ctc_lens.cpu(), b["ctc_lens"]).to(device)
                else:
                    loss_ctc = ctc(ctc_lp.transpose(0, 1).float(), b["ctc"].to(device), ctc_lens, b["ctc_lens"].to(device))
            loss = loss_ce + args.ctc_weight * loss_ctc
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if log and step % 50 == 0:
                log({"step": step, "lr": sched.get_last_lr()[0], "loss": loss.item(), "ce": loss_ce.item(),
                     "ctc": loss_ctc.item(), "sec": round(time.time() - t0, 1)})
        n = int((tout != PAD_ID).sum())
        tot["loss"] += loss.item() * n
        tot["ce"] += loss_ce.item() * n
        tot["ctc"] += loss_ctc.item() * n
        tot["tokens"] += n
        tot["batches"] += 1
    d = max(tot["tokens"], 1)
    return {"loss": tot["loss"] / d, "ce": tot["ce"] / d, "ctc": tot["ctc"] / d, "ppl": math.exp(min(tot["ce"] / d, 20)),
            "batches": tot["batches"], "sec": round(time.time() - t0, 1)}, step


def main(argv=None) -> None:
    args = build_args(argv)
    seed_everything(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"

    def log(obj):
        obj = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **obj}
        print(json.dumps(obj), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    prep_cfg = json.loads((args.prepared / "prep_config.json").read_text())
    if prep_cfg["feature_version"] != FEATURE_VERSION:
        sys.exit(f"Prepared data uses feature version {prep_cfg['feature_version']}, code is {FEATURE_VERSION}")
    lm_root = Path(prep_cfg["landmarks_root"])
    vocab = Vocab.load(args.prepared / "vocab.json")
    stats = json.loads((args.prepared / "feature_stats.json").read_text())

    train_rows = load_manifest(args.prepared / f"{args.train_manifest}.jsonl")
    val_rows = load_manifest(args.prepared / f"{args.val_manifest}.jsonl")
    if {r["name"] for r in train_rows} & {r["name"] for r in val_rows}:
        sys.exit("Train and validation manifests share examples.")
    if {r["video_id"] for r in train_rows} & {r["video_id"] for r in val_rows}:
        sys.exit("Train and validation manifests share source videos.")
    skipped_long = sum(r["frames"] > args.max_frames for r in train_rows)
    train_rows = [r for r in train_rows if r["frames"] <= args.max_frames]
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]

    train_ds = SignTextDataset(train_rows, lm_root, vocab, train=True, seed=args.seed, ctc_source=args.ctc_source)
    val_ds = SignTextDataset(val_rows, lm_root, vocab, train=False, ctc_source=args.ctc_source)
    train_sampler = BucketSampler([r["frames"] for r in train_rows], args.batch_size, shuffle=True, seed=args.seed)
    val_sampler = BucketSampler([r["frames"] for r in val_rows], args.batch_size, shuffle=False)
    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, **kw)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, **kw)

    mcfg = ModelConfig(in_dim=FEATURE_DIM, vocab_size=len(vocab), d_model=args.d_model, enc_layers=args.enc_layers,
                       dec_layers=args.dec_layers, heads=args.heads, ff=args.ff, dropout=args.dropout)
    model = SLTModel(mcfg)
    model.encoder.set_stats(stats["mean"], stats["std"])
    fresh = not (args.resume and (args.out / "last.pt").exists())
    if args.init_model and fresh:
        pre = torch.load(args.init_model, map_location="cpu", weights_only=False)
        if pre["meta"]["feature_version"] != FEATURE_VERSION:
            sys.exit("The model to fine-tune uses a different feature version.")
        if pre["config"]["vocab_size"] != len(vocab):
            sys.exit(f"--init-model has a {pre['config']['vocab_size']}-word vocabulary but --prepared has {len(vocab)}. "
                     "Copy the model's vocab.json into the prepared folder.")
        model.load_state_dict(pre["model"], strict=True)  # includes the feature statistics it was trained with
        log({"event": "init_model", "from": str(args.init_model)})
    elif args.init_encoder and fresh:
        pre = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
        if pre["meta"]["feature_version"] != FEATURE_VERSION:
            sys.exit("The pre-trained encoder uses a different feature version.")
        enc_state = {k[len("encoder."):]: v for k, v in pre["model"].items() if k.startswith("encoder.")}
        # The pre-trained weights expect inputs normalised with the statistics they were
        # trained with, so those come along (they are baked into the exported encoder).
        model.encoder.load_state_dict(enc_state, strict=True)  # also loads feat_mean / feat_std
        log({"event": "init_encoder", "from": str(args.init_encoder), "pretrain_best_val_top1": pre.get("best")})
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = max(1, args.epochs * len(train_sampler))
    warm = min(args.warmup_steps, max(1, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    start_epoch, step, best, bad = 0, 0, float("inf"), 0
    last = args.out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng"])
        start_epoch, step, best, bad = ck["epoch"] + 1, ck["step"], ck["best"], ck["bad"]
        log({"event": "resumed", "epoch": start_epoch, "step": step})

    n_params = sum(p.numel() for p in model.parameters())
    meta = {"model": mcfg.to_dict(), "feature_version": FEATURE_VERSION, "target_fps": TARGET_FPS,
            "train_manifest": args.train_manifest, "val_manifest": args.val_manifest,
            "train_examples": len(train_rows), "val_examples": len(val_rows), "skipped_long": skipped_long,
            "ctc_source": args.ctc_source, "init_encoder": str(args.init_encoder) if args.init_encoder else None, "init_model": str(args.init_model) if args.init_model else None, "args": {k: str(v) for k, v in vars(args).items()}, "params": n_params,
            "device": str(device), "torch": torch.__version__}
    (args.out / "run_config.json").write_text(json.dumps(meta, indent=2))
    log({"event": "start", "device": str(device), "params": n_params, "train": len(train_rows),
         "val": len(val_rows), "skipped_long": skipped_long})

    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.hours and epoch > start_epoch:
            per_epoch = (time.time() - run_start) / (epoch - start_epoch)
            if time.time() - run_start + per_epoch > args.hours * 3600:
                log({"event": "time_limit", "epoch": epoch, "hours": args.hours})
                break
        train_sampler.epoch = epoch
        train_ds.epoch = epoch
        tr, step = run_epoch(model, train_dl, device, opt, sched, args, log, step)
        va, _ = run_epoch(model, val_dl, device, args=args)
        improved = va["ce"] < best
        if improved:
            best, bad = va["ce"], 0
        else:
            bad += 1
        log({"epoch": epoch, "train": tr, "val": va, "best_val_ce": best, "improved": improved})
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "rng": torch.get_rng_state(), "epoch": epoch, "step": step, "best": best, "bad": bad,
              "config": mcfg.to_dict(), "meta": meta}
        torch.save(ck, last)
        if improved:
            torch.save(ck, args.out / "best.pt")
        if bad >= args.patience:
            log({"event": "early_stop", "epoch": epoch})
            break
    log({"event": "done", "best_val_ce": best})


if __name__ == "__main__":
    main()
