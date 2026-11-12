"""Readers for the real How2Sign re-aligned English translation files.

Format verified against the downloaded files (see ml/README.md, "Dataset"):
tab-separated, header
  VIDEO_ID  VIDEO_NAME  SENTENCE_ID  SENTENCE_NAME  START_REALIGNED  END_REALIGNED  SENTENCE
SENTENCE_NAME == f"{SENTENCE_ID}-{n}-rgb_front", and the sentence-level clip file is
f"{SENTENCE_NAME}.mp4". `n` takes values 1..11 matching the signer numbers on
how2sign.github.io; we use it as the signer id. That mapping is our inference
and is not documented by the dataset authors.

The published release contains NO gloss annotations (the download script lists
"Gloss annotations" under TODO). An optional gloss file can be supplied with
columns SENTENCE_NAME and GLOSS; nothing in this repo invents one.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
EXPECTED_COLUMNS = ["VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
                    "START_REALIGNED", "END_REALIGNED", "SENTENCE"]


@dataclass
class Record:
    split: str
    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    start: float
    end: float
    sentence: str
    signer: str
    gloss: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def annotation_path(root: Path, split: str) -> Path:
    return Path(root) / "annotations" / f"how2sign_realigned_{split}.csv"


def read_split(root: Path, split: str) -> list[Record]:
    path = annotation_path(root, split)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}, expected {EXPECTED_COLUMNS}")
        out = []
        for r in reader:
            signer = r["VIDEO_NAME"].rsplit("-", 2)[1]
            out.append(Record(split, r["VIDEO_ID"], r["VIDEO_NAME"], r["SENTENCE_ID"], r["SENTENCE_NAME"],
                              float(r["START_REALIGNED"]), float(r["END_REALIGNED"]), r["SENTENCE"].strip(), signer))
    return out


def read_glosses(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "SENTENCE_NAME" not in reader.fieldnames or "GLOSS" not in reader.fieldnames:
            raise ValueError(f"{path}: gloss file needs tab-separated SENTENCE_NAME and GLOSS columns")
        return {r["SENTENCE_NAME"]: r["GLOSS"].strip() for r in reader}
