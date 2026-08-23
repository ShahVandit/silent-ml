"""Determinism utilities.

Every training run in this environment is keyed by an integer seed so that a
given (pipeline, operator, params, seed) reproduces identical metrics — the
foundation the judge relies on for causal ablation and reproducible rewards.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs and force deterministic algorithms where feasible.

    CPU-deterministic. On CUDA, callers should also keep batch sizes and data
    order fixed; the environment does so by construction.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seeding CUDA creates a CUDA context, which costs a few hundred MB and
    # fails outright on a GPU another job has filled. Skip it unless CUDA is
    # actually going to be used, so a CPU run never touches the device.
    if os.environ.get("SILENTML_DEVICE", "").startswith("cuda") or (
        not os.environ.get("SILENTML_DEVICE") and torch.cuda.is_available()
    ):
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass   # GPU unusable; the pipeline falls back to CPU anyway
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding so multi-worker loading stays reproducible."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
