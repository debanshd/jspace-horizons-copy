#!/usr/bin/env python
"""The name without the fact, across sizes: D = P(the receiver names the donor's object and answers the animal rule for its own).
    venv/bin/python jbound/fig_f4.py --runs runs/run4/Qwen3.5-0.8B runs/run4/Qwen3.5-2B runs/run4/Qwen3.5-4B runs/run4/Qwen3.5-27B runs/run4/gemma-4-31B-it --out runs/run4/figF4_sizes.png
Left: every arm on the first run given (the 4B). Right: the four arms that matter, one column per model.
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

ARM_COL = {"full state, read layers": OKABE[6], "lens part, read layers": OKABE[0], "J++ part, read layers": OKABE[2], "complement of the lens part, read layers": OKABE[1],
           "random part, read layers (mean of 3)": OKABE[4], "placebo donor (empty canvas), read layers": "0.6", "full state, every layer": "0.35", "lens part, every layer": OKABE[0],
           "random part, every layer": OKABE[4], "non-target donor, full state, every layer": "0.75", "the fact typed as text": OKABE[5], "receiver alone": "0.85"}
SHORT = {"full state, read layers": "the whole state, middle layers", "lens part, read layers": "the lens part, middle layers", "J++ part, read layers": "the J++ part, middle layers",
         "complement of the lens part, read layers": "the complement of the lens part, middle layers", "random part, read layers (mean of 3)": "a random part, middle layers (3 draws)",
         "placebo donor (empty canvas), read layers": "a donor that saw an empty canvas", "full state, every layer": "the whole state, every layer", "lens part, every layer": "the lens part, every layer",
         "random part, every layer": "a random part, every layer", "non-target donor, full state, every layer": "a donor with the object in another slot", "the fact typed as text": "the fact typed as text", "receiver alone": "nothing moved"}
KEY = ["lens part, read layers", "complement of the lens part, read layers", "full state, read layers", "random part, read layers (mean of 3)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {os.path.basename(r.rstrip("/")): json.load(open(f"{r}/eF4_results.json")) for r in a.runs if os.path.exists(f"{r}/eF4_results.json")}
    names = list(R)
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.75))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.2, top=0.8)
    first = R[names[0]]
    arms = [k for k in SHORT if k in first["D"]]
    y = np.arange(len(arms))[::-1]
    v = [first["D"][k] for k in arms]
    ax[0].barh(y, [t[0] for t in v], 0.62, color=[ARM_COL[k] for k in arms])
    ax[0].errorbar([t[0] for t in v], y, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].set(yticks=y, yticklabels=[SHORT[k] for k in arms], xlim=(0, 1.02), xlabel=f"share of scenes: names the donor's object and keeps its own animal answer ({first['n']} scenes, 95% CI)")
    ax[0].set_title(f"{names[0]}, fresh crops, rank {first['rank']}", fontsize=9.5)
    x = np.arange(len(names))
    for j, k in enumerate(KEY):
        vals = [R[n]["D"].get(k, (np.nan, np.nan, np.nan)) for n in names]
        off = (j - 1.5) * 0.19
        ax[1].errorbar(x + off, [t[0] for t in vals], yerr=[[t[0] - t[1] for t in vals], [t[2] - t[0] for t in vals]], fmt="o", ms=5, color=ARM_COL[k], ecolor=ARM_COL[k], elinewidth=0.8, capsize=0, label=SHORT[k], lw=0)
    ax[1].set(xticks=x, xticklabels=names, ylim=(-0.02, 1.02), ylabel="the same share, by model")
    ax[1].tick_params(axis="x", labelsize=8, rotation=20)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=7.5, ncol=2)
    ax[1].set_title("the four arms that decide it, one column per model", fontsize=9.5)
    d = first["D"]
    title = (f"The name without the fact. Moved at the middle layers into a receiver that sees its own object, the lens part of a donor's picture state makes the receiver name the donor's object and keep its own animal answer on "
             f"{d['lens part, read layers'][0]:.2f} of scenes ({names[0]}); the complement {d['complement of the lens part, read layers'][0]:.2f}, the whole state {d['full state, read layers'][0]:.2f}, a random part {d['random part, read layers (mean of 3)'][0]:.2f}, an empty-canvas donor {d['placebo donor (empty canvas), read layers'][0]:.2f}.")
    fig.suptitle("\n".join(textwrap.wrap(title, 135)), fontsize=9.5, y=0.97)
    fig.text(0.005, 0.005, "F4 · fresh crops (COCO ids in eF4_results.json) · first-word and first-character scoring · 95% scene bootstrap", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
