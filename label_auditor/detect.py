"""Label-issue detectors. Each returns (flags, score); higher score = more likely mislabeled."""
from __future__ import annotations

import numpy as np


def self_confidence(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability the model gives to the label the example was given."""
    return probs[np.arange(len(y)), y]


def argmax_disagree(probs: np.ndarray, y: np.ndarray):
    """Baseline: flag when the model's top class differs from the given label."""
    return probs.argmax(1) != y, 1 - self_confidence(probs, y)


def confident_learning(probs: np.ndarray, y: np.ndarray):
    """Confident learning (Northcutt et al., 2021), written from scratch.

    Per-class threshold t_j = mean predicted p(j) over examples labeled j. An example is
    "confidently" class j if p(j) >= t_j (ties broken by the highest probability). Examples
    whose confident class differs from their label sit off the diagonal of the confident
    joint and are flagged. Classes the model is generally unsure of get lower thresholds,
    so this adapts to per-class difficulty where a single cut-off cannot.
    """
    k = probs.shape[1]
    thresholds = np.array([probs[y == j, j].mean() if (y == j).any() else 1.0 for j in range(k)])
    above = probs >= thresholds - 1e-12
    masked = np.where(above, probs, -1.0)
    confident = masked.argmax(1)
    has_conf = above.any(1)
    flags = has_conf & (confident != y)
    return flags, 1 - self_confidence(probs, y)


def ensemble_vote(per_model: dict[str, np.ndarray], y: np.ndarray):
    """Model disagreement: flag when most models' top class differs from the label."""
    wrong = np.stack([p.argmax(1) != y for p in per_model.values()])
    frac = wrong.mean(0)
    mean_conf = np.mean([self_confidence(p, y) for p in per_model.values()], axis=0)
    return frac > 0.5, frac + (1 - mean_conf)  # vote share first, confidence breaks ties


def cleanlab_issues(probs: np.ndarray, y: np.ndarray):
    """Reference implementation, used to check ours."""
    from cleanlab.filter import find_label_issues

    flags = find_label_issues(labels=y, pred_probs=probs, filter_by="prune_by_noise_rate")
    return flags, 1 - self_confidence(probs, y)


def average(per_model: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(list(per_model.values()), axis=0)


def run_all(per_model: dict[str, np.ndarray], y: np.ndarray, with_cleanlab: bool = True) -> dict:
    avg = average(per_model)
    out = {
        "argmax": argmax_disagree(avg, y),
        "confident_learning": confident_learning(avg, y),
        "ensemble_vote": ensemble_vote(per_model, y),
    }
    if with_cleanlab:
        try:
            out["cleanlab"] = cleanlab_issues(avg, y)
        except ImportError:
            pass
    return out
