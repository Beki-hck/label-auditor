"""Score a detector against the labels we know were flipped."""
from __future__ import annotations

import numpy as np


def prf(flags: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((flags & truth).sum())
    p = tp / flags.sum() if flags.sum() else 0.0
    r = tp / truth.sum() if truth.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "flagged": int(flags.sum())}


def precision_at_k(score: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    """Of the k highest-scored examples, how many were really flipped (k defaults to the true count)."""
    k = int(truth.sum()) if k is None else k
    if k == 0:
        return 0.0
    top = np.argsort(-score, kind="stable")[:k]
    return float(truth[top].mean())
