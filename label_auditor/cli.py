"""Command line: `audit` a dataset for suspect labels, or `bench` detectors on injected noise."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import data, detect, metrics, probs


def cmd_audit(a) -> None:
    ds = data.load(a.data)
    per_model = probs.all_models(ds.X, ds.y, a.models, a.folds)
    avg = detect.average(per_model)
    flags, score = detect.run_all(per_model, ds.y, with_cleanlab=False)[a.method]
    order = np.argsort(-score, kind="stable")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "given_label", "suggested_label", "p_given", "p_suggested", "flagged"])
        for rank, i in enumerate(order[: a.top], 1):
            s = int(avg[i].argmax())
            w.writerow([rank, ds.ids[i], ds.class_names[ds.y[i]], ds.class_names[s],
                        f"{avg[i, ds.y[i]]:.3f}", f"{avg[i, s]:.3f}", bool(flags[i])])
    print(f"{ds.name}: {int(flags.sum())} of {len(ds.y)} flagged by {a.method}; top {a.top} written to {out}")


def bench(spec: str, rates, kinds, seeds, models, folds) -> list[dict]:
    ds = data.load(spec)
    k = len(ds.class_names)
    rows = []
    for kind in kinds:
        for rate in rates:
            for seed in seeds:
                noisy, truth = data.inject_noise(ds.y, rate, k, seed, kind)
                per_model = probs.all_models(ds.X, noisy, models, folds)
                for name, (flags, score) in detect.run_all(per_model, noisy).items():
                    rows.append({"dataset": ds.name, "noise": kind, "rate": rate, "seed": seed,
                                 "method": name, **metrics.prf(flags, truth),
                                 "p_at_k": metrics.precision_at_k(score, truth)})
    return rows


METHOD_ORDER = ["argmax", "ensemble_vote", "confident_learning", "cleanlab"]


def summarize(rows: list[dict]) -> str:
    keys = sorted({(r["dataset"], r["noise"], r["rate"], r["method"]) for r in rows},
                  key=lambda t: (t[0], t[1], t[2], METHOD_ORDER.index(t[3])))
    lines = ["| dataset | noise | rate | method | precision | recall | F1 | P@k |",
             "|---|---|---|---|---|---|---|---|"]
    for key in keys:
        g = [r for r in rows if (r["dataset"], r["noise"], r["rate"], r["method"]) == key]
        m = {f: np.mean([r[f] for r in g]) for f in ("precision", "recall", "f1", "p_at_k")}
        lines.append(f"| {key[0]} | {key[1]} | {key[2]:.0%} | {key[3]} | {m['precision']:.2f} | "
                     f"{m['recall']:.2f} | {m['f1']:.2f} | {m['p_at_k']:.2f} |")
    return "\n".join(lines)


def cmd_bench(a) -> None:
    rows = []
    for spec in a.data:
        rows += bench(spec, a.rates, a.kinds, range(a.seeds), a.models, a.folds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    table = summarize(rows)
    (out / "results.md").write_text(table + "\n")
    print(table)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="label-auditor")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", nargs="+", default=list(probs.MODELS), choices=list(probs.MODELS))
    common.add_argument("--folds", type=int, default=5)

    au = sub.add_parser("audit", parents=[common], help="rank examples by how likely their label is wrong")
    au.add_argument("data", help="iris | digits | csv:PATH | folder:PATH")
    au.add_argument("--method", default="confident_learning",
                    choices=["argmax", "confident_learning", "ensemble_vote"])
    au.add_argument("--top", type=int, default=50)
    au.add_argument("-o", "--out", default="audit.csv")
    au.set_defaults(fn=cmd_audit)

    be = sub.add_parser("bench", parents=[common], help="inject known label noise and score each detector")
    be.add_argument("data", nargs="+")
    be.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    be.add_argument("--kinds", nargs="+", default=["uniform", "pair"], choices=["uniform", "pair"])
    be.add_argument("--seeds", type=int, default=3)
    be.add_argument("-o", "--out", default="results")
    be.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
