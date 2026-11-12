"""Device selection and seeding. On the Mac: Apple MPS, falling back to CPU.
CUDA is used only where it exists, i.e. the optional Kaggle notebook run
(see README, "Training on Kaggle"); the app itself never needs a GPU."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA requested but not available; using CPU")
    if requested == "mps":
        print("[device] MPS requested but not available; using CPU")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
