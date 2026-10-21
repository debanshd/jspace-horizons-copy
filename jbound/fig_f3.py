#!/usr/bin/env python
"""The transplant figure: a part of a donor's picture state moved into a receiver, scored on the first letter (the name) and on the animal
rule (the fact). Left: F1, the receiver's slot is empty (runs/run2b). Right: F3, the receiver's slot holds its own object of the other kind.
    venv/bin/python jbound/fig_f3.py --f1 runs/run2b/Qwen3.5-4B --f3 runs/run3/Qwen3.5-4B --out runs/run3/figF3_transplants_Qwen3.5-4B.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

BLUE, ORANGE = OKABE[0], OKABE[1]
ROWS_F1 = [("blank, no transplant", "nothing moved"), ("full state, read layers", "the full state, read layers"), ("J part, read layers", "the lens part, read layers"),
           ("random part, read layers", "a random part, read layers"), ("full state, every layer", "the full state, every layer"), ("J part, every layer", "the lens part, every layer"),
           ("random part, every layer", "a random part, every layer"), ("the fact typed as text", "the fact typed as text")]
ROWS_F3 = [("receiver alone", "nothing moved"), ("full state, read layers", "the full state, read layers"), ("lens part, read layers", "the lens part, read layers"),
           ("J++ part, read layers", "the J++ part, read layers"), ("random part, read layers", "a random part, read layers (3 draws)"),
           ("complement of the lens part, read layers", "the complement of the lens part, read layers"), ("full state, every layer", "the full state, every layer"),
           ("lens part, every layer", "the lens part, every layer"), ("random part, every layer", "a random part, every layer"),
           ("non-target donor, full state, every layer", "a donor with the object elsewhere"), ("the fact typed as text", "the fact typed as text")]


def panel(ax, rows, get, n, title, xlabel):
    rows = [(k, lab) for k, lab in rows if get("letter", k) is not None]
    y = np.arange(len(rows))[::-1]
    for j, (task, c, lab) in enumerate((("letter", BLUE, "gives its first letter"), ("animal", ORANGE, "applies the animal rule"))):
        v = [get(task, k) for k, _ in rows]
        ax.barh(y + (0.5 - j) * 0.38, [t[0] for t in v], 0.36, color=c, alpha=0.9, label=lab)
        ax.errorbar([t[0] for t in v], y + (0.5 - j) * 0.38, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax.set(yticks=y, yticklabels=[lab for _, lab in rows], xlim=(0, 1.02), xlabel=xlabel)
    ax.set_title(title, fontsize=9.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1", required=True)
    ap.add_argument("--f3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r1 = json.load(open(f"{a.f1}/eF1_results.json"))
    r3 = json.load(open(f"{a.f3}/eF3_results.json"))
    name = os.path.basename(a.f3.rstrip("/"))
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.9))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.2, top=0.8)

    def g1(task, k):
        return r1["rules"][task]["accuracy"].get(k)

    def g3(task, k):
        c = r3["carried"][task]
        if k == "random part, read layers":
            draws = [c[d] for d in c if d.startswith("random part, read layers, draw")]
            return (float(np.mean([d[0] for d in draws])), float(np.mean([d[1] for d in draws])), float(np.mean([d[2] for d in draws]))) if draws else None
        return c.get(k)

    panel(ax[0], ROWS_F1, g1, r1["n"], "receiver with an empty slot", f"accuracy on a rule the donor never saw (95% CI, {r1['n']} scenes)")
    panel(ax[1], ROWS_F3, g3, r3["n"], "receiver with its own object in the slot", f"share of answers that follow the donor's object (95% CI, {r3['n']} scenes)")
    ax[0].axvline(0.5, color="0.7", lw=0.7, ls=":")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, ncol=2)
    c3 = r3["carried"]
    rnd = g3("animal", "random part, read layers")
    title = (f"{name}: moved into a receiver that sees its own object, either half of the donor's middle-layer state carries the donor's name (the lens part {c3['say']['lens part, read layers'][0]:.2f}, "
             f"its complement {c3['say']['complement of the lens part, read layers'][0]:.2f}) and neither carries the animal rule ({c3['animal']['lens part, read layers'][0]:.2f}, {c3['animal']['complement of the lens part, read layers'][0]:.2f}); "
             f"the whole state carries both ({c3['say']['full state, read layers'][0]:.2f}, {c3['animal']['full state, read layers'][0]:.2f}); a random part neither ({g3('say', 'random part, read layers')[0]:.2f}, {rnd[0]:.2f}).")
    fig.suptitle("\n".join(textwrap.wrap(title, 125)), fontsize=9.5, y=0.97)
    fig.text(0.005, 0.005, f"F1 ({os.path.basename(a.f1.rstrip('/'))}) and F3 · first-character scoring · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
