#!/usr/bin/env python
"""Fit each depth profile with a falling logistic step, R(x) = a / (1 + exp((x - c) / tau)), x = the moved window's first layer as a share
of depth, and bootstrap c (the horizon) over scenes. Reads d1_depth.csv from each run; writes horizons.json beside it and prints a table.
    venv/bin/python jbound/fit_horizons.py --runs runs/run4/Qwen3.5-4B runs/run4/Qwen3.5-2B runs/run4/Qwen3.5-0.8B
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

KINDS = ["transplant", "transplant printed", "transplant typed"]


def step(x, a, c, tau):
    return a / (1.0 + np.exp((x - c) / max(tau, 1e-3)))


def fit(xs, ys):
    try:
        p, _ = curve_fit(step, xs, ys, p0=[max(ys.max(), 0.05), 0.5, 0.05], bounds=([0, 0, 0.005], [1.05, 1.2, 0.5]), maxfev=4000)
        return p
    except Exception:
        return np.array([np.nan] * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for run in a.runs:
        f = f"{run}/d1_depth.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        r = json.load(open(f"{run}/eD1_results.json"))
        n_layers = r["n_layers"]
        name = os.path.basename(run.rstrip("/"))
        out = {}
        scenes = np.array(sorted(d.scene.unique()))
        for kind in KINDS:
            dk = d[d.kind == kind]
            if dk.empty:
                continue
            for task in sorted(dk.task.unique()):
                dt = dk[dk.task == task]
                piv = dt.pivot_table(index="scene", columns="w0", values="follows_X", aggfunc="mean").reindex(scenes)
                xs = (piv.columns.values + 1) / n_layers
                ys = piv.values.astype(float)
                p = fit(xs, np.nanmean(ys, 0))
                bs = []
                for _ in range(a.boot):
                    idx = rng.integers(0, len(scenes), len(scenes))
                    bs.append(fit(xs, np.nanmean(ys[idx], 0))[1])
                bs = np.array(bs)
                lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
                out[f"{kind} | {task}"] = dict(a=float(p[0]), c=float(p[1]), tau=float(p[2]), c_lo=float(lo), c_hi=float(hi), top=float(np.nanmean(ys, 0).max()))
                rows.append((name, kind, task, p[1], lo, hi, p[2], float(np.nanmean(ys, 0).max())))
        json.dump(out, open(f"{run}/horizons.json", "w"), indent=1)
    print(f"{'model':14s} {'data type':20s} {'property':8s} {'horizon c':>10s} {'95% CI':>16s} {'tau':>6s} {'top':>5s}")
    for name, kind, task, c, lo, hi, tau, top in rows:
        print(f"{name:14s} {kind:20s} {task:8s} {c:10.2f} [{lo:5.2f}, {hi:5.2f}] {tau:6.3f} {top:5.2f}")
    for name in sorted({r[0] for r in rows}):
        for kind in KINDS:
            cs = {r[2]: r[3] for r in rows if r[0] == name and r[1] == kind}
            if "say" in cs and "animal" in cs and np.isfinite(cs["say"]) and np.isfinite(cs["animal"]):
                print(f"{name} {kind}: gap c_say - c_animal = {cs['say'] - cs['animal']:.2f} of depth")


if __name__ == "__main__":
    main()
