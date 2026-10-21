#!/usr/bin/env python
"""The paradox and its mechanism. Left: the category is decodable from the donor's state at every depth (grey) while the model reads it
only up to the use horizon (orange, from D1). Right: what the receiver does with the donor's state by where it is delivered (D2).
    venv/bin/python jbound/fig_paradox.py --d1 runs/run4/Qwen3.5-4B --d2 runs/run5/Qwen3.5-4B --out runs/run5/figParadox_Qwen3.5-4B.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

BLUE, ORANGE = OKABE[0], OKABE[1]
ARMS = [("donor's early-window state in place (control)", "the donor's early state, delivered early"), ("donor in the band only", "the donor's band state, delivered in the band"),
        ("donor's band state delivered in the early window", "the donor's band state, delivered early"), ("donor early, receiver restored in the band", "the donor's early state, the receiver's own restored in the band")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", required=True)
    ap.add_argument("--d2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="picture")
    a = ap.parse_args()
    d1 = json.load(open(f"{a.d1}/eD1_results.json")); d2 = json.load(open(f"{a.d2}/eD2_results.json"))
    name = os.path.basename(a.d2.rstrip("/"))
    n = d1["n_layers"]
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.95))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.8)
    dec = d2["decodable_by_layer"]
    xs_d = [(int(L) + 1) / n for L in dec]; ys_d = [dec[L] for L in dec]
    ax[0].plot(xs_d, ys_d, color="0.45", lw=2.0, solid_capstyle="round")
    xs = [(w + 1) / n for w in d1["windows"]]
    use = d1["curves"]["transplant"]["animal"]
    ax[0].fill_between(xs, [v[1] for v in use], [v[2] for v in use], color=ORANGE, alpha=0.16, lw=0)
    ax[0].plot(xs, [v[0] for v in use], color=ORANGE, lw=2.0, marker="o", ms=4, solid_capstyle="round")
    ax[0].axvspan(0.33, 0.8, color="0.93", lw=0, zorder=0)
    ax[0].set(xlim=(0, 1), ylim=(-0.02, 1.02), xlabel="depth, as a share of layers", ylabel="share of scenes")
    ax[0].text(0.03, 0.80, "the category is decodable\nfrom the state at every depth", color="0.4", fontsize=8.5, ha="left", va="top", transform=ax[0].transAxes)
    ax[0].text(0.50, 0.42, "the animal rule follows the\nstate delivered at this depth", color=ORANGE, fontsize=8.5, ha="left", va="top", transform=ax[0].transAxes)
    ax[0].set_title("present at every depth, read only early", fontsize=9.5)
    arms = d2["arms"][a.source]
    rows = [(lab, arms[k]) for k, lab in ARMS if k in arms]
    y = np.arange(len(rows))[::-1]
    for j, (key, c, lab) in enumerate((("say_X", BLUE, "names the donor's object"), ("animal_X", ORANGE, "applies the animal rule to the donor's object"))):
        v = [r[key] for _, r in rows]
        ax[1].barh(y + (0.5 - j) * 0.4, [t[0] for t in v], 0.38, color=c, alpha=0.85)
        ax[1].errorbar([t[0] for t in v], y + (0.5 - j) * 0.4, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        fig.text(0.60 + 0.2 * j, 0.045, lab, color=c, fontsize=8.5, ha="left")
    ax[1].set(yticks=y, yticklabels=[lab for lab, _ in rows], xlim=(0, 1.0), xlabel=f"share of {d2['n']} scenes")
    ax[1].tick_params(axis="y", labelsize=8)
    ax[1].set_title("the same state, delivered at a different depth", fontsize=9.5)
    t = arms.get("donor's band state delivered in the early window", {}); b = arms.get("donor in the band only", {})
    title = (f"{name}: the donor's band state delivered in the band is named ({b.get('say_X', [0])[0]:.2f}) and not used ({b.get('animal_X', [0])[0]:.2f}); the same state delivered early is used ({t.get('animal_X', [0])[0]:.2f})."
             if t and b else f"{name}: the paradox and its mechanism")
    fig.suptitle("\n".join(textwrap.wrap(title, 125)), fontsize=9.5, y=0.97)
    fig.text(0.005, 0.005, f"D1 and D2 · {d2['n']} fresh scenes · leave-one-out nearest-centroid probe on the mean-pooled slot state · 95% scene bootstrap · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
