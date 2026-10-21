#!/usr/bin/env python
"""The depth law (D1) and the load law (K1), one figure each, in the suite's style.
    venv/bin/python jbound/fig_depth.py --d1 runs/run4/Qwen3.5-4B runs/run4/Qwen3.5-2B --k1 runs/run4/Qwen3.5-4B --out-dir runs/run4
figDepth_<models>.png: one panel per model; x = the first layer of the moved window as a share of depth; y = the share of scenes on which
the receiver names the donor's object (blue) or applies the animal rule to it (orange); solid = the picture donor's state moved, dashed =
the printed word's, dotted = the typed sentence's; the thin grey line is the injected name direction (naming only).
figLoad_<model>.png: the cost of removing the object's own name direction, in nats, against the number of objects on the canvas.
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

BLUE, ORANGE = OKABE[0], OKABE[1]
KINDS = (("transplant", "-", "the picture's state"), ("transplant printed", "--", "the printed word's state"), ("transplant typed", ":", "the typed sentence's state"))


def depth(runs, out):
    runs = [r.rstrip("/") for r in runs]
    R = {os.path.basename(r.rstrip("/")): json.load(open(f"{r}/eD1_results.json")) for r in runs if os.path.exists(f"{r}/eD1_results.json")}
    names = list(R)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + 1.2, 4.6), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.2, top=0.78, wspace=0.18)
    cuts = {}
    for ax, name in zip(axes[0], names):
        r = R[name]
        n_layers = r["n_layers"]
        xs = [(w + 1) / n_layers for w in r["windows"]]
        ends = []
        for kind, ls, lab in KINDS:
            if kind not in r["curves"]:
                continue
            for task, c in (("say", BLUE), ("animal", ORANGE)):
                ys = r["curves"][kind][task]
                ax.plot(xs, [v[0] for v in ys], ls=ls, color=c, lw=2.0 if kind == "transplant" else 1.4, solid_capstyle="round")
                if kind == "transplant":
                    ax.fill_between(xs, [v[1] for v in ys], [v[2] for v in ys], color=c, alpha=0.16, lw=0)
        inj = r["curves"].get("inject", {}).get("say")
        if inj:
            ax.plot(xs, [v[0] for v in inj], color="0.6", lw=1.0)
        try:
            lay = json.load(open(os.path.join(os.path.dirname(runs[names.index(name)].rstrip("/")) if False else runs[names.index(name)], "config.json")))["layers"]
            ax.axvspan((lay[0] + 1) / n_layers, (lay[-1] + 1) / n_layers, color="0.92", lw=0, zorder=0)
        except Exception:
            pass
        cut = r["cut"].get("transplant", {})
        cuts[name] = cut
        ax.set(xlim=(0, 1), ylim=(-0.02, 1.02), xlabel="first layer of the moved window, as a share of depth", title=f"{name}: {cut.get('verdict', '')}")
        if cut.get("L_say") is not None:
            ax.axvline((cut["L_say"] + 1) / n_layers, color=BLUE, lw=0.8, ls=":")
        if cut.get("L_use") is not None:
            ax.axvline((cut["L_use"] + 1) / n_layers, color=ORANGE, lw=0.8, ls=":")
    axes[0][0].set_ylabel("share of scenes that follow the donor's object")
    ax = axes[0][-1]
    ax.text(1.02, 0.95, "names the donor's object", color=BLUE, transform=ax.transAxes, fontsize=8.5, va="top")
    ax.text(1.02, 0.86, "applies the animal rule to it", color=ORANGE, transform=ax.transAxes, fontsize=8.5, va="top")
    ax.text(1.02, 0.77, "the name direction injected\n(naming only, grey)", color="0.45", transform=ax.transAxes, fontsize=8, va="top")
    ax.text(1.02, 0.60, "solid: the picture's state\ndashed: the printed word's\ndotted: the typed sentence's\n\ngrey band: the layers the\ntext lens is read at", color="0.3", transform=ax.transAxes, fontsize=8, va="top")
    parts = []
    for name in names:
        try:
            hz = json.load(open(os.path.join(runs[names.index(name)], "horizons.json")))
            parts.append(f"{name}: fact {hz['transplant | animal']['c']:.2f}, name {hz['transplant | say']['c']:.2f}")
        except Exception:
            parts.append(name)
    fig.suptitle("\n".join(textwrap.wrap("A donor's state moved into a receiver one window of four layers at a time: the fact stops moving before the name in every model. Fitted horizons as a share of depth: " + "; ".join(parts) + ".", 150)), fontsize=9.5, y=0.98)
    fig.text(0.005, 0.005, "D1 · 64 fresh scenes per model · bands 95% scene bootstrap on the picture donor · dotted verticals: L_say (blue) and L_use (orange)", fontsize=6.5, color="0.5")
    figstyle.finish(fig, out)


def load(run, out):
    r = json.load(open(f"{run}/eK1_results.json"))
    name = os.path.basename(run.rstrip("/"))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    fig.subplots_adjust(left=0.15, right=0.8, bottom=0.17, top=0.8)
    ks = r["loads"]
    ends = []
    for task, c, lab in (("say", BLUE, "naming the object"), ("animal", ORANGE, "the animal rule")):
        v = [r["cost"][f"k={k} | {task} | own name"] for k in ks]
        ax.fill_between(ks, [t[1] for t in v], [t[2] for t in v], color=c, alpha=0.16, lw=0)
        ax.plot(ks, [t[0] for t in v], "-", marker="o", ms=4.5, lw=2.0, color=c, solid_capstyle="round")
        ends.append((v[-1][0], lab, c))
    for arm, ls in (("other name", "--"), ("random direction", ":")):
        v = [r["cost"][f"k={k} | say | {arm}"][0] for k in ks]
        ax.plot(ks, v, ls=ls, color="0.55", lw=1.0)
        ends.append((v[-1], f"{arm} removed, naming", "0.45"))
    for y, (y0, lab, c) in zip([t[0] for t in figstyle.spread([(e[0], e) for e in ends], 0.06, *ax.get_ylim())], [t[1] for t in figstyle.spread([(e[0], e) for e in ends], 0.06, *ax.get_ylim())]):
        ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), va="center", ha="left", color=c, fontsize=8.5, clip_on=False)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xticks=ks, xlabel="objects on the canvas", ylabel="change in log p of the correct answer, nats")
    d = r["k6_minus_k1_say"]
    fig.suptitle("\n".join(textwrap.wrap(f"{name}: removing the object's own name direction costs {-r['cost']['k=1 | say | own name'][0]:.2f} nats on naming with one object and {-r['cost']['k=6 | say | own name'][0]:.2f} with six (difference {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]): {r['verdict']}", 95)), fontsize=9.5, y=0.98)
    fig.text(0.005, 0.005, f"K1 · {r['n']} scenes · mean-ablation on the slot's tokens at the read layers · 95% scene bootstrap · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1", nargs="*", default=[])
    ap.add_argument("--k1", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    if a.d1:
        depth(a.d1, f"{a.out_dir}/figDepth_{'_'.join(os.path.basename(r.rstrip('/')) for r in a.d1)}.png")
    for run in a.k1:
        load(run, f"{a.out_dir}/figLoad_{os.path.basename(run.rstrip('/'))}.png")


if __name__ == "__main__":
    main()
