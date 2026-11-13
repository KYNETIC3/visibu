"""Evaluate a checkpoint on one split. Never evaluates on the training manifest.

Reports, on the chosen split only:
  translation  BLEU-1..4 and chrF (sacrebleu), ROUGE-L (rouge-score)
  CTC head     WER of greedy CTC output against glosses, or against English words
               when no glosses exist (then it is an auxiliary-head diagnostic, NOT
               sign recognition accuracy)
  per signer   BLEU-4 per signer; signers absent from training are marked unseen
  suggestions  oracle BLEU of the best of the app's n-best suggestions (how much
               picking a suggestion could help); beam search unless --beam 1
  runtime      CPU latency per sentence (batch 1), parameter count / fp32 size
  examples     hypotheses next to references, best/worst/random

  python -m training.evaluate --checkpoint checkpoints/slt-base/best.pt --split val
  python -m training.evaluate --checkpoint checkpoints/slt-base/best.pt --split test   # final, once
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from common.device import pick_device  # noqa: E402
from common.tokenizer import BOS_ID, EOS_ID, PAD_ID, PUNCT, UNK_ID, Vocab, tokenize  # noqa: E402
from models.beam import BeamConfig  # noqa: E402
from models.ctc_head import ctc_greedy_decode  # noqa: E402
from models.slt import ModelConfig, SLTModel  # noqa: E402
from training.data import BucketSampler, SignTextDataset, collate, load_manifest  # noqa: E402


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


def load_model(path: Path) -> tuple[SLTModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SLTModel(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def main() -> None:
    import sacrebleu
    from rouge_score import rouge_scorer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--prepared", type=Path, default=ML_ROOT / "datasets" / "how2sign" / "prepared")
    ap.add_argument("--split", required=True, help="manifest name, e.g. val, test, smoke_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--latency-samples", type=int, default=30)
    ap.add_argument("--beam", type=int, default=4, help="beam width; 1 = greedy (as in the app before suggestions)")
    ap.add_argument("--n-best", type=int, default=3, help="suggestions per sentence, as shown in the app")
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--out", type=Path, help="default: reports/eval_<run>_<split>.json")
    args = ap.parse_args()

    model, ck = load_model(args.checkpoint)
    meta = ck["meta"]
    if args.split == meta["train_manifest"]:
        sys.exit("Refusing to evaluate on the training manifest.")
    device = pick_device(args.device)
    model.to(device)
    prep = json.loads((args.prepared / "prep_config.json").read_text())
    vocab = Vocab.load(args.prepared / "vocab.json")
    rows = load_manifest(args.prepared / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    train_signers = {r["signer"] for r in load_manifest(args.prepared / f"{meta['train_manifest']}.jsonl")}
    ds = SignTextDataset(rows, Path(prep["landmarks_root"]), vocab, ctc_source=meta.get("ctc_source", "auto"))
    dl = DataLoader(ds, batch_sampler=BucketSampler([r["frames"] for r in rows], args.batch_size, False),
                    collate_fn=collate, num_workers=2)

    bcfg = BeamConfig(beam=args.beam, n_best=args.n_best, no_repeat_ngram=args.no_repeat_ngram,
                      banned=(PAD_ID, BOS_ID, UNK_ID))
    hyps, refs, signers, confs, names, nbests = [], [], [], [], [], []
    ctc_err, ctc_n = 0, 0
    for b in dl:
        feats, lens = b["feats"].to(device), b["feat_lens"].to(device)
        ids, conf, memory, _ = model.greedy(feats, lens, BOS_ID, EOS_ID)
        if args.beam > 1:
            out_lens = model.encoder.out_lengths(lens).tolist()
            ids, conf = [], []
            for j, n in enumerate(out_lens):
                hs = model.beam(memory[j:j + 1, :n], BOS_ID, EOS_ID, bcfg)
                ids.append(hs[0].tokens if hs else [])
                conf.append(float(torch.tensor(hs[0].score).exp()) if hs else 0.0)
                nbests.append([vocab.decode(h.tokens) for h in hs])
        else:
            nbests += [[vocab.decode(x)] for x in ids]
        hyps += [vocab.decode(x) for x in ids]
        refs += b["sentences"]
        signers += b["signers"]
        confs += conf
        names += b["names"]
        ctc_hyp = ctc_greedy_decode(model.ctc(memory), model.encoder.out_lengths(lens))
        off = 0
        for j, n in enumerate(b["ctc_lens"].tolist()):
            ref_ids = b["ctc"][off:off + n].tolist()
            off += n
            ctc_err += edit_distance(ctc_hyp[j], ref_ids)
            ctc_n += max(n, 1)

    def lower_words(s: str) -> str:
        return " ".join(t for t in tokenize(s))

    H = [lower_words(h) for h in hyps]
    R = [lower_words(r) for r in refs]
    bleu = sacrebleu.corpus_bleu(H, [R], tokenize="none")
    chrf = sacrebleu.corpus_chrf(H, [R])
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    rouge_l = st.mean(scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(H, R))
    word_wer = sum(edit_distance([t for t in h.split() if t not in PUNCT], [t for t in r.split() if t not in PUNCT])
                   for h, r in zip(H, R)) / max(1, sum(len([t for t in r.split() if t not in PUNCT]) for r in R))

    # Oracle: if the signer always picked the closest of the suggestions shown.
    oracle = []
    for cands, r in zip(nbests, R):
        cs = [lower_words(c) for c in cands] or [""]
        oracle.append(max(cs, key=lambda c: sacrebleu.sentence_bleu(c, [r], tokenize="none").score))
    oracle_bleu = sacrebleu.corpus_bleu(oracle, [R], tokenize="none")

    per_signer = {}
    for s in sorted(set(signers), key=int):
        idx = [i for i, x in enumerate(signers) if x == s]
        per_signer[s] = {"n": len(idx), "unseen_in_training": s not in train_signers,
                         "bleu4": round(sacrebleu.corpus_bleu([H[i] for i in idx], [[R[i] for i in idx]],
                                                              tokenize="none").score, 2)}

    # CPU latency, batch 1, the way the browser runs it (encoder once, then beam or greedy decoding).
    cpu_model = model.to("cpu")
    lat = []
    for i in random.Random(0).sample(range(len(ds)), min(args.latency_samples, len(ds))):
        x = ds[i]["feats"][None]
        t0 = time.perf_counter()
        if args.beam > 1:
            with torch.no_grad():
                mem, _ = cpu_model.encoder(x, None)
            cpu_model.beam(mem, BOS_ID, EOS_ID, bcfg)
        else:
            cpu_model.greedy(x, torch.tensor([x.shape[1]]), BOS_ID, EOS_ID)
        lat.append((time.perf_counter() - t0) * 1000)
    n_params = sum(p.numel() for p in model.parameters())

    order = sorted(range(len(H)), key=lambda i: sacrebleu.sentence_bleu(H[i], [R[i]], tokenize="none").score)
    pick = lambda idx: [{"name": names[i], "signer": signers[i], "confidence": round(confs[i], 3),
                         "hypothesis": hyps[i], "reference": refs[i]} for i in idx]
    report = {
        "checkpoint": str(args.checkpoint), "split": args.split, "examples": len(H),
        "note": "Test-split numbers are final results; do not tune on them." if args.split == "test" else
                "Validation/held-out numbers, used for model selection.",
        "translation": {"bleu": round(bleu.score, 2), "bleu_precisions_1to4": [round(p, 2) for p in bleu.precisions],
                        "chrf": round(chrf.score, 2), "rougeL_f": round(rouge_l * 100, 2),
                        "word_error_rate": round(word_wer, 4)},
        "decoding": {"beam": args.beam, "n_best": args.n_best, "no_repeat_ngram": args.no_repeat_ngram},
        "suggestions": {"oracle_bleu_best_of_n": round(oracle_bleu.score, 2),
                        "note": "Upper bound: BLEU if the signer always picked the closest suggestion."},
        "ctc_head": {"wer": round(ctc_err / max(ctc_n, 1), 4),
                     "targets": "glosses" if meta.get("ctc_source") == "gloss" else
                                "English words (auxiliary diagnostic, not sign recognition accuracy)"},
        "confidence": {"mean": round(st.mean(confs), 3)},
        "per_signer": per_signer,
        "runtime_cpu": {"latency_ms_median": round(st.median(lat), 1), "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
                        "params": n_params, "fp32_megabytes": round(n_params * 4 / 1e6, 1), "torch_threads": torch.get_num_threads()},
        "samples": {"best": pick(order[-5:][::-1]), "worst": pick(order[:5]),
                     "random": pick(random.Random(1).sample(range(len(H)), min(10, len(H))))},
        "train_meta": meta,
    }
    out = args.out or ML_ROOT / "reports" / f"eval_{args.checkpoint.parent.name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("split", "examples", "decoding", "translation", "suggestions", "ctc_head", "confidence",
                                             "per_signer", "runtime_cpu")}, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
