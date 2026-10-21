#!/usr/bin/env python
"""The horizon spectrum: for each property, the last window start (as a share of depth) at which moving the donor's state still makes
the property follow the donor, one dot per model and data type.
    venv/bin/python jbound/fig_spectrum.py --runs runs/run5/Qwen3.5-4B runs/run5/Qwen3.5-2B --out runs/run5/figSpectrum.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

ORDER = ["animal", "ride", "set", "letter", "say"]
LABEL = {"animal": "animal rule (referent)", "ride": "ride rule (function)", "set": "set rule (name, symbolic)", "letter": "first letter (name, lexical)", "say": "the name itself"}
KIND = {"transplant": ("o", "the picture's state"), "transplant printed": ("s", "the printed word's"), "transplant typed": ("^", "the typed sentence's")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {os.path.basename(r.rstrip("/")): json.load(open(f"{r}/eD1_results.json")) for r in a.runs if os.path.exists(f"{r}/eD1_results.json")}
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.3, right=0.8, bottom=0.16, top=0.8)
    cols = [OKABE[0], OKABE[1], OKABE[2], OKABE[3], OKABE[4]]
    y = np.arange(len(ORDER))[::-1]
    for mi, (name, r) in enumerate(R.items()):
        n = r["n_layers"]; W = r["windows"]
        for kind, (mk, lab) in KIND.items():
            if kind not in r["curves"]:
                continue
            for pi, prop in enumerate(ORDER):
                if prop not in r["curves"][kind]:
                    continue
                ok = [w for w, v in zip(W, r["curves"][kind][prop]) if v[0] >= 0.5]
                if not ok:
                    continue
                x = (max(ok) + 1 + (W[1] - W[0])) / n            # the end of the last window that still moves the property, as a share of depth
                ax.plot([x], [y[pi] + (mi - (len(R) - 1) / 2) * 0.18], marker=mk, ms=6, color=cols[mi % len(cols)], lw=0, alpha=0.9)
    ax.set(yticks=y, yticklabels=[LABEL[p] for p in ORDER], xlim=(0, 1.02), xlabel="depth up to which a moved state still sets the property (share of layers)")
    for mi, name in enumerate(R):
        ax.text(1.02, 0.95 - 0.07 * mi, name, color=cols[mi % len(cols)], transform=ax.transAxes, fontsize=8.5, va="top")
    for ki, (kind, (mk, lab)) in enumerate(KIND.items()):
        ax.plot([], [], marker=mk, color="0.4", lw=0, label=lab)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle("\n".join(textwrap.wrap("The horizon spectrum: properties defined over the referent stop being movable early, properties defined over the name late", 95)), fontsize=9.5, y=0.98)
    fig.text(0.005, 0.005, "D1 with the spectrum questions · 64 fresh scenes per model · windows of 4 layers · threshold 0.5 of scenes", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
