"""Word-level English tokeniser. Mirrored by src/ml/tokenizer.ts (detokenise only).

Word level (not BPE) because it needs no extra runtime in the browser and the
How2Sign test-set out-of-vocabulary rate with a frequency>=3 vocabulary is low
(measured by inspect_dataset.py).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
CTC_BLANK_ID = PAD_ID  # CTC blank shares the padding id; pad never appears in targets.

PUNCT = set(".,!?;:")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[.,!?;:]")


def normalize(text: str) -> str:
    return (text.lower().replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"'))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def detokenize(tokens: list[str]) -> str:
    out = ""
    cap_next = True
    for t in tokens:
        if t in SPECIALS:
            continue
        word = t
        if word == "i" or word.startswith("i'"):
            word = "I" + word[1:]
        if cap_next and word not in PUNCT:
            word = word[:1].upper() + word[1:]
            cap_next = False
        if word in PUNCT:
            out += word
            if word in ".!?":
                cap_next = True
        else:
            out += (" " if out else "") + word
    return out


class Vocab:
    def __init__(self, itos: list[str]):
        assert itos[:4] == SPECIALS
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts, min_freq: int = 3, max_size: int = 8000) -> "Vocab":
        c = Counter(t for text in texts for t in tokenize(text))
        words = [w for w, n in sorted(c.items(), key=lambda x: (-x[1], x[0])) if n >= min_freq]
        return cls(SPECIALS + words[: max_size - len(SPECIALS)])

    def encode(self, text: str, add_bos_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)]
        return [BOS_ID] + ids + [EOS_ID] if add_bos_eos else ids

    def ctc_targets(self, text: str) -> list[int]:
        """Word ids for the auxiliary CTC head: no punctuation, no unknown words."""
        return [self.stoi[t] for t in tokenize(text) if t not in PUNCT and t in self.stoi]

    def decode(self, ids: list[int]) -> str:
        toks = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            toks.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return detokenize(toks)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text())["itos"])
