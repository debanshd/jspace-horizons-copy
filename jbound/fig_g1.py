#!/usr/bin/env python
"""G1 report figures from a run folder (eG1_results.json), one message per plot.
    venv/bin/python jbound/fig_g1.py --run runs/run3/Qwen3.5-4B --out-dir runs/run3
writes figG1_seen_or_said_<model>.png (the other name injected on the picture or at the answer: dose curves and the matched-dose bars)
and figG1_handle_removed_<model>.png (change in log p of the correct answer when one part of the slot's tokens is removed).
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

ALPHAS = [1, 2, 4, 8]
BLUE, ORANGE, GREEN, PINK, YELLOW, SKY = OKABE[0], OKABE[1], OKABE[2], OKABE[3], OKABE[4], OKABE[5]
SHORT = {"own name": "the object's own name direction", "other name": "another object's name direction", "random direction": "a random direction",
         "own name, W_U only": "own name, output-layer direction only", "own name, J++": "own name, J++ direction", "the word 'animal'": "the word 'animal'",
         "own name, orthogonal to 'animal'": "own name, orthogonal to 'animal'", "top principal direction of the slot": "the slot's top principal direction",
         "lens part, 256 directions": "the lens part, 256 directions", "random part, 256 directions": "a random part, 256 directions"}
SITE = {"image tokens": "on the object's tokens", "all image tokens": "on every image token", "decision position": "at the answer position"}


def ci(v):
    return [[v[0] - v[1]], [v[2] - v[0]]]


def seen_or_said(r, name, out, picture_site):
    acc, fol, sites = r["accuracy"], r["follow"], r["sites"]
    if sites[picture_site].get("alpha_star") is None and sites.get("all image tokens", {}).get("alpha_star") is not None:
        picture_site = "all image tokens"
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.8)
    xs = np.arange(len(ALPHAS) + 1)
    for site, ls in ((picture_site, "-"), ("decision position", "--")):
        blk = f"inject at the {site}"
        for task, c, mk, lab in (("say", BLUE, "o", "names it"), ("animal", ORANGE, "s", "applies the animal rule to it")):
            ys = [fol["own picture | clean | 0"][task]] + [fol[f"{blk} | other name | {a}"][task] for a in ALPHAS]
            ax[0].plot(xs, [y[0] for y in ys], ls=ls, marker=mk, ms=4.5, color=c, lw=1.4, label=f"{lab}, {SITE[site]}")
            ax[0].fill_between(xs, [y[1] for y in ys], [y[2] for y in ys], color=c, alpha=0.1, lw=0)
        yr = [fol["own picture | clean | 0"]["say"][0]] + [fol[f"{blk} | random direction | {a}"]["say"][0] for a in ALPHAS]
        ax[0].plot(xs, yr, ls=ls, color="0.65", lw=0.9, label=f"random direction, {SITE[site]}")
    ax[0].set(xticks=xs, xticklabels=["0"] + [str(a) for a in ALPHAS], xlabel="dose of the other object's name direction (units of a token's own-name component)",
              ylabel="share of answers that follow the injected name", ylim=(-0.02, 1.02))
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=2)
    ax[0].set_title("the other object's name, written into the model's state", fontsize=9.5)
    rows = []
    for site in (picture_site, "decision position"):
        o = sites[site]
        if o.get("alpha_star") is None:
            continue
        rows += [(f"other name, {SITE[site]} (dose {o['alpha_star']:g})", o["rules_follow_Y"], o["say_follows_Y"], ORANGE),
                 (f"same-kind name, {SITE[site]}", o["Z_rules_flip"], o["Z_say_follows_Z"], GREEN),
                 (f"random direction, {SITE[site]}", o["rules_follow_random"], None, YELLOW)]
    if rows:
        y = np.arange(len(rows))[::-1]
        ax[1].barh(y, [t[1][0] for t in rows], 0.62, color=[t[3] for t in rows], alpha=0.9)
        ax[1].errorbar([t[1][0] for t in rows], y, xerr=[[t[1][0] - t[1][1] for t in rows], [t[1][2] - t[1][0] for t in rows]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        for yi, t in zip(y, rows):
            if t[2] is not None:
                ax[1].plot([t[2][0]], [yi], marker="o", ms=5, color=BLUE, mec="white", mew=0.6, zorder=5)
        ax[1].set(yticks=y, yticklabels=[t[0] for t in rows], xlim=(0, 1.02), xlabel=f"share of rule answers that change to the injected object's kind ({r['n_held']} held-out scenes)")
        ax[1].plot([], [], marker="o", color=BLUE, lw=0, label="names the injected object")
        ax[1].legend(loc="lower right", fontsize=7.5)
    ax[1].set_title("at the dose that makes the model name it", fontsize=9.5)
    pic, dec = sites[picture_site], sites["decision position"]
    f4 = fol[f"inject at the {picture_site} | other name | 4"]
    r4 = fol[f"inject at the {picture_site} | random direction | 4"]
    w4 = fol[f"inject at the {picture_site} | other name, W_U only | 4"]
    title = (f"{name}: the other object's name direction, written {SITE[picture_site]} at dose 4, is said on {f4['say'][0]:.2f} of scenes and spelled on {f4['letter'][0]:.2f}; "
             f"the animal rule keeps following the picture ({1 - f4['animal'][0]:.2f}). A random direction {r4['say'][0]:.2f}, the output-layer direction {w4['say'][0]:.2f}.")
    fig.suptitle("\n".join(textwrap.wrap(title, 120)), fontsize=9.5, y=0.97)
    fig.text(0.005, 0.005, f"G1 · {r['n']} scenes, dose picked on {r['n'] - r['n_held']} and read on {r['n_held']} · bands and bars 95% scene bootstrap · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, out)


def handle_removed(r, name, out):
    dl = r["dlogp"]
    arms = [a for a in SHORT if f"own picture | {a}" in dl]
    rows = [(SHORT[a], dl[f"own picture | {a}"]) for a in arms] + [("printed word instead of the photo: own name direction", dl["printed word | own name"])]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 2.2))
    fig.subplots_adjust(left=0.42, right=0.97, bottom=0.17, top=0.8)
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= 0.7
    for j, (task, c, lab) in enumerate((("say", BLUE, "names it"), ("animal", ORANGE, "applies the animal rule"))):
        v = [t[1][task] for t in rows]
        ax.barh(y + (0.5 - j) * 0.38, [x[0] for x in v], 0.36, color=c, alpha=0.9, label=lab)
        ax.errorbar([x[0] for x in v], y + (0.5 - j) * 0.38, xerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set(yticks=y, yticklabels=["removed: " + t[0] for t in rows], xlabel=f"change in log p of the correct answer, nats (95% CI over {r['n']} scenes)")
    ax.legend(loc="upper right", fontsize=8)
    own, oth = dl["own picture | own name"], dl["own picture | other name"]
    title = (f"{name}: removing the object's own name direction from its 49 tokens costs {-own['say'][0]:.2f} nats to name it and {-own['animal'][0]:.2f} on the animal rule; "
             f"another object's name direction costs {-oth['say'][0]:.2f} and {-oth['animal'][0]:.2f}")
    fig.suptitle("\n".join(textwrap.wrap(title, 110)), fontsize=9.5, y=0.98)
    fig.text(0.005, 0.005, f"G1 · mean-ablation at the read layers on the slot's tokens · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--picture-site", default="image tokens")
    a = ap.parse_args()
    r = json.load(open(f"{a.run}/eG1_results.json"))
    name = os.path.basename(a.run.rstrip("/"))
    figstyle.apply(titlesize=9.5, labelsize=8.5)
    seen_or_said(r, name, f"{a.out_dir}/figG1_seen_or_said_{name}.png", a.picture_site)
    handle_removed(r, name, f"{a.out_dir}/figG1_handle_removed_{name}.png")


if __name__ == "__main__":
    main()
