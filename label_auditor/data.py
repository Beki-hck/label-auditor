"""Datasets and synthetic label noise."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]
    ids: list[str]


def load(spec: str) -> Dataset:
    """`iris`, `digits`, `csv:PATH` (last column = label) or `folder:PATH` (one subfolder per class)."""
    if spec in ("iris", "digits"):
        from sklearn import datasets

        d = datasets.load_iris() if spec == "iris" else datasets.load_digits()
        names = [str(n) for n in d.target_names]
        return Dataset(spec, d.data.astype(float), d.target.astype(int), names,
                       [f"{spec}-{i}" for i in range(len(d.target))])
    kind, _, path = spec.partition(":")
    if kind == "csv":
        return _load_csv(Path(path))
    if kind == "folder":
        from .embed import embed_folder

        return embed_folder(Path(path))
    raise ValueError(f"unknown dataset spec: {spec}")


def _load_csv(path: Path) -> Dataset:
    import csv

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    body = rows[1:]
    labels = [r[-1] for r in body]
    names = sorted(set(labels))
    y = np.array([names.index(lab) for lab in labels])
    X = np.array([[float(v) for v in r[:-1]] for r in body])
    return Dataset(path.stem, X, y, names, [f"row-{i + 2}" for i in range(len(body))])


def inject_noise(y: np.ndarray, rate: float, n_classes: int, seed: int = 0,
                 kind: str = "uniform") -> tuple[np.ndarray, np.ndarray]:
    """Flip `rate` of labels. uniform: to any other class. pair: class k -> k+1 (harder, systematic).

    Returns (noisy_labels, flipped_mask).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.choice(n, size=int(round(rate * n)), replace=False)
    noisy = y.copy()
    for i in idx:
        if kind == "pair":
            noisy[i] = (y[i] + 1) % n_classes
        else:
            noisy[i] = rng.choice([c for c in range(n_classes) if c != y[i]])
    return noisy, noisy != y
