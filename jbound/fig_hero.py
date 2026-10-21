#!/usr/bin/env python
"""The headline figure. Left (bars): moved into a receiver that sees its own object, which parts of a donor's picture state carry the
donor's name, and which carry the donor's fact into the animal rule. Right (two lines): another object's name direction written onto the
picture at rising dose: the model says it, the animal rule ignores it.
    venv/bin/python jbound/fig_hero.py --transplant runs/run4/Qwen3.5-4B/eF4_results.json --inject runs/run3/Qwen3.5-4B/eG1_results.json --out runs/run4/figHero_Qwen3.5-4B.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

BLUE, ORANGE = OKABE[0], OKABE[1]
ROWS = [("receiver alone", "nothing moved"), ("random part, read layers", "a random part, 256 of 2560 directions"), ("lens part, read layers", "the lens part, 256 directions"),
        ("complement of the lens part, read layers", "its complement, 2304 directions"), ("full state, read layers", "the whole state")]


def get(r, task, arm):
    c = r["carried"][task]
    if arm == "random part, read layers":
        d = [c[k] for k in c if k.startswith("random part, read layers, draw")]
        return tuple(float(np.mean([t[i] for t in d])) for i in range(3))
    return tuple(c[arm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transplant", required=True)
    ap.add_argument("--inject", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="all image tokens")
    a = ap.parse_args()
    T, G = json.load(open(a.transplant)), json.load(open(a.inject))
    name = os.path.basename(os.path.dirname(a.transplant))
    figstyle.apply(titlesize=10, labelsize=9)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.5))
    fig.subplots_adjust(left=0.19, right=0.84, bottom=0.15, top=0.8)
    y = np.arange(len(ROWS))
    for j, (task, c, lab) in enumerate((("say", BLUE, "names the donor's object"), ("animal", ORANGE, "applies the animal rule to the donor's object"))):
        v = [get(T, task, k) for k, _ in ROWS]
        ax[0].barh(y + (0.5 - j) * 0.4, [t[0] for t in v], 0.38, color=c, alpha=0.85)
        ax[0].errorbar([t[0] for t in v], y + (0.5 - j) * 0.4, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        ax[0].text(0.98, 0.97 - 0.07 * j, lab, color=c, fontsize=8.5, transform=ax[0].transAxes, va="top", ha="right")
    ax[0].set(yticks=y, yticklabels=[lab for _, lab in ROWS], xlim=(0, 1.0), xlabel="share of scenes")
    ax[0].invert_yaxis()
    ax[0].set_title("a part of a donor's picture state, moved into a receiver that sees a different object", fontsize=9.5)
    xs = np.arange(5)
    fol = G["follow"]
    blk = f"inject at the {a.site}"
    ends = []
    for task, c, lab in (("say", BLUE, "says the injected name"), ("animal", ORANGE, "applies the animal rule to it")):
        ys = [fol["own picture | clean | 0"][task]] + [fol[f"{blk} | other name | {d}"][task] for d in (1, 2, 4, 8)]
        ax[1].fill_between(xs, [t[1] for t in ys], [t[2] for t in ys], color=c, alpha=0.16, lw=0)
        ax[1].plot(xs, [t[0] for t in ys], "-", marker="o", ms=4.5, lw=2.0, color=c, solid_capstyle="round")
        ends.append((ys[-1][0], lab, c))
    yr = [fol["own picture | clean | 0"]["say"][0]] + [fol[f"{blk} | random direction | {d}"]["say"][0] for d in (1, 2, 4, 8)]
    ax[1].plot(xs, yr, lw=1.2, color="0.55", ls="--")
    ends.append((yr[-1], "a random direction, says it", "0.45"))
    for yy, (y0, lab, c) in zip([t[0] for t in figstyle.spread([(e[0], e) for e in ends], 0.075, 0.0, 0.98)], [t[1] for t in figstyle.spread([(e[0], e) for e in ends], 0.075, 0.0, 0.98)]):
        ax[1].text(1.02, yy, lab, transform=ax[1].get_yaxis_transform(), va="center", ha="left", color=c, fontsize=8.5, clip_on=False)
    ax[1].set(xticks=xs, xticklabels=["0", "1", "2", "4", "8"], xlabel="dose of the other object's name direction (units of a token's own-name component)", ylim=(-0.02, 1.0), ylabel="share of answers that follow the injected name")
    ax[1].set_title("another object's name direction, written onto every image token", fontsize=9.5)
    s4 = fol[f"{blk} | other name | 4"]
    tl, tc, tf = get(T, "say", "lens part, read layers"), get(T, "say", "complement of the lens part, read layers"), get(T, "say", "full state, read layers")
    al, ac, af = get(T, "animal", "lens part, read layers"), get(T, "animal", "complement of the lens part, read layers"), get(T, "animal", "full state, read layers")
    title = (f"{name}: either half of the picture state carries the object's name ({tl[0]:.2f}, {tc[0]:.2f}) and neither carries the fact ({al[0]:.2f}, {ac[0]:.2f}); the whole state carries both ({tf[0]:.2f}, {af[0]:.2f}). "
             f"A name written onto the picture is said ({s4['say'][0]:.2f} at dose 4) and the animal rule ignores it ({s4['animal'][0]:.2f}).")
    fig.suptitle("\n".join(textwrap.wrap(title, 125)), fontsize=10, y=0.98)
    fig.text(0.005, 0.005, f"left: {T['n']} scenes, middle layers, 95% scene bootstrap · right: {G['n']} scenes, 95% scene bootstrap · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
