import numpy as np
import pytest

from label_auditor import data, detect, metrics
from label_auditor.cli import main


def test_inject_noise_flips_exact_count_and_never_keeps_label():
    y = np.repeat(np.arange(3), 50)
    noisy, flipped = data.inject_noise(y, 0.2, 3, seed=1)
    assert flipped.sum() == 30
    assert (noisy[flipped] != y[flipped]).all()
    assert (noisy[~flipped] == y[~flipped]).all()


def test_pair_noise_moves_to_next_class():
    y = np.repeat(np.arange(3), 10)
    noisy, flipped = data.inject_noise(y, 0.5, 3, seed=0, kind="pair")
    assert ((y[flipped] + 1) % 3 == noisy[flipped]).all()


def test_noise_is_reproducible():
    y = np.repeat(np.arange(4), 25)
    assert (data.inject_noise(y, 0.1, 4, 7)[0] == data.inject_noise(y, 0.1, 4, 7)[0]).all()


def test_confident_learning_flags_obvious_flip_only():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.95, 0.05]])
    y = np.array([0, 0, 1, 1, 1])  # last one looks like class 0
    flags, score = detect.confident_learning(probs, y)
    assert flags.tolist() == [False, False, False, False, True]
    assert score.argmax() == 4


def test_confident_learning_uses_per_class_thresholds():
    # class 1 is "hard": the model is never very sure of it, so 0.45 still counts as confident
    probs = np.array([[0.95, 0.05], [0.9, 0.1], [0.55, 0.45], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 1, 0])
    flags, _ = detect.confident_learning(probs, y)
    assert flags[4]  # labeled 0 but confidently class 1 by class 1's threshold
    assert not flags[2]


def test_ensemble_vote_needs_majority():
    y = np.array([0, 0])
    per = {"a": np.array([[0.2, 0.8], [0.9, 0.1]]),
           "b": np.array([[0.3, 0.7], [0.4, 0.6]]),
           "c": np.array([[0.9, 0.1], [0.9, 0.1]])}
    flags, _ = detect.ensemble_vote(per, y)
    assert flags.tolist() == [True, False]


def test_metrics():
    flags = np.array([1, 1, 0, 0], bool)
    truth = np.array([1, 0, 1, 0], bool)
    m = metrics.prf(flags, truth)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert metrics.precision_at_k(np.array([0.9, 0.1, 0.8, 0.0]), truth) == 1.0


def test_csv_loader(tmp_path):
    f = tmp_path / "toy.csv"
    f.write_text("a,b,label\n1,2,cat\n3,4,dog\n5,6,cat\n")
    ds = data.load(f"csv:{f}")
    assert ds.class_names == ["cat", "dog"] and ds.y.tolist() == [0, 1, 0]
    assert ds.X.shape == (3, 2)


def test_bench_finds_injected_noise_on_iris(tmp_path, capsys):
    main(["bench", "iris", "--rates", "0.1", "--kinds", "uniform", "--seeds", "1",
          "--models", "logreg", "--folds", "3", "-o", str(tmp_path)])
    out = (tmp_path / "results.md").read_text()
    assert "confident_learning" in out
    import json
    rows = json.loads((tmp_path / "results.json").read_text())
    cl = next(r for r in rows if r["method"] == "confident_learning")
    assert cl["recall"] > 0.5 and cl["precision"] > 0.5


def test_audit_writes_ranked_csv(tmp_path):
    out = tmp_path / "a.csv"
    main(["audit", "iris", "--models", "logreg", "--top", "5", "-o", str(out)])
    lines = out.read_text().splitlines()
    assert lines[0].startswith("rank,id") and len(lines) == 6


def test_unknown_dataset():
    with pytest.raises(ValueError):
        data.load("nope")
