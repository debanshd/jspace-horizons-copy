#!/usr/bin/env python
"""T1 figure from a run's t1_transport.csv: the arms by rendering (zero- and mean-ablation), the mean-ablated relative cost by rendering
and load, and the slot-localized arms. Relative cost = (random − J) / (clean − no evidence), scene bootstrap; a relative ablation cost,
not a fraction of information carried.
    venv/bin/python jbound/fig_t1.py --run runs/t1pilot2/Qwen3.5-4B --out runs/t1pilot2/figT1_Qwen3.5-4B.png
"""
import argparse, os, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE, spread

RENDS = ["image", "printed", "text"]
RC = {"image": OKABE[0], "printed": OKABE[2], "text": OKABE[5]}
ARMS = [("no evidence", "no evidence", "0.4"), ("ablate J part r=256", "J-lens part", OKABE[0]), ("ablate J++ part r=256", "J++ part", OKABE[2]),
        ("ablate W_U part r=256", "W_U part (no Jacobian)", OKABE[5]), ("ablate PCA part r=256", "top-256 PCs", OKABE[3]), ("ablate random part r=256", "random part", OKABE[4])]


def d(df, arm):
    c = df[df.arm == "clean"].set_index("scene").logp
    x = df[df.arm == arm].set_index("scene").logp
    return x - c.loc[x.index]


def boot(frames, f, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = frames[0].index.values
    vals = [f(*[fr.loc[idx[s]].values for fr in frames]) for s in rng.integers(0, len(idx), (n, len(idx)))]
    return float(f(*[fr.values for fr in frames])), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pd.read_csv(f"{a.run}/t1_transport.csv")
    name = os.path.basename(a.run.rstrip("/"))
    ks = sorted(t.k.unique())
    kmax = ks[-1]
    n = t.scene.nunique()
    figstyle.apply(titlesize=9, labelsize=8.5)
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.4), gridspec_kw=dict(wspace=0.38, width_ratios=[1.7, 1, 1]))
    fig.subplots_adjust(bottom=0.18, top=0.8, left=0.05, right=0.99)
    # A: arms by rendering at kmax, zero-ablation solid and mean-ablation hatched
    arms = [(k_, lab, c) for k_, lab, c in ARMS if k_ in set(t.arm)]
    bars = []
    for key, lab, c in arms:
        bars.append((key, lab, c, ""))
        mk = key.replace("ablate ", "mean-ablate ")
        if key != "no evidence" and mk in set(t.arm):
            bars.append((mk, lab + ", mean-ablated", c, "xx"))
    w = 0.84 / len(bars)
    for j, (key, lab, c, h) in enumerate(bars):
        v = [boot([d(t[(t.k == kmax) & (t.modality == m)], key)], lambda x: x.mean()) for m in RENDS]
        xs = np.arange(3) + (j - len(bars) / 2 + 0.5) * w
        ax[0].bar(xs, [q[0] for q in v], w, color=c, alpha=0.9 if not h else 0.45, hatch=h, lw=0)
        ax[0].errorbar(xs, [q[0] for q in v], yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ntok = {m: t[(t.modality == m)].shape[0] for m in RENDS}
    ax[0].set(xticks=range(3), xticklabels=[f"{m}" for m in RENDS], xlabel=f"the same {n} scenes, k = {kmax} items, one single-fact question; arms at the read layers on every evidence token")
    ax[0].set_ylabel("Δ log p(target noun) vs clean (scene 95% CI)")
    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(lo, hi + (0.1 + 0.05 * len(bars)) * (hi - lo))
    for j, (key, lab, c, h) in enumerate(bars):
        ax[0].text(0.02 + 0.5 * (j >= (len(bars) + 1) // 2), 0.97 - 0.055 * (j % ((len(bars) + 1) // 2)), lab, color=c, fontsize=7.5, transform=ax[0].transAxes, va="top", alpha=1 if not h else 0.7)
    # B: mean-ablated relative cost by rendering and load
    rel = {}
    for m in RENDS:
        for k_ in ks:
            df = t[(t.k == k_) & (t.modality == m)]
            arm = "mean-ablate J part r=256" if "mean-ablate J part r=256" in set(df.arm) else "ablate J part r=256"
            rarm = arm.replace("J part", "random part")
            ev, j_, r_ = -d(df, "no evidence"), d(df, arm), d(df, rarm)
            rel[(m, k_)] = (boot([j_, r_, ev], lambda x, y, e: (y - x).mean() / e.mean()), (j_, r_, ev))
    mean_used = "mean-ablate J part r=256" in set(t.arm)
    ends = []
    for m in RENDS:
        v = [rel[(m, k_)][0] for k_ in ks]
        ax[1].errorbar(ks, [q[0] for q in v], yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], fmt="-o", color=RC[m], ms=4, lw=1.6, capsize=3)
        ends.append((v[-1][0], m))
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].axhline(1, color="0.6", lw=0.8, ls=":")
    ax[1].set(xlabel="items in the scene (k)", xticks=ks, xlim=(ks[0] - 0.6, ks[-1] + 1.9))
    ax[1].set_ylabel(("mean-ablated " if mean_used else "") + "relative cost of the J-lens part\n(random − J) / (clean − no evidence), scene 95% CI")
    ylo, yhi = ax[1].get_ylim()
    ax[1].set_ylim(min(ylo, -0.02), max(yhi, 1.08))
    ylo, yhi = ax[1].get_ylim()
    for y, m in spread(ends, 0.05 * (yhi - ylo), ylo + 0.03 * (yhi - ylo), yhi - 0.03 * (yhi - ylo)):
        ax[1].text(ks[-1] + 0.25, y, m, color=RC[m], fontsize=8.5, va="center")
    ax[1].text(ks[-1] + 1.85, 1.0, "1 = as costly as\nremoving the evidence", color="0.45", fontsize=6.5, va="bottom", ha="right")
    # pairwise contrasts at kmax
    pairs = {}
    for p, q in (("printed", "text"), ("image", "text"), ("printed", "image")):
        (ja, ra, ea), (jb, rb, eb) = rel[(p, kmax)][1], rel[(q, kmax)][1]
        pairs[(p, q)] = boot([ja, ra, ea, jb, rb, eb], lambda x1, x2, x3, y1, y2, y3: (x2 - x1).mean() / x3.mean() - (y2 - y1).mean() / y3.mean())
    # C: slot-localized arms at kmax
    snames = [("ablate J part r=256, target slot", "J part, target slot", OKABE[0], ""), ("ablate random part r=256, target slot", "random part, target slot", OKABE[4], ""),
              ("ablate J part r=256, other slot", "J part, other slot", OKABE[0], "//"), ("ablate random part r=256, other slot", "random part, other slot", OKABE[4], "//")]
    pix = [m for m in RENDS if m != "text"]
    w = 0.8 / len(snames)
    for j, (key, lab, c, h) in enumerate(snames):
        v = [boot([d(t[(t.k == kmax) & (t.modality == m)], key)], lambda x: x.mean()) for m in pix]
        xs = np.arange(len(pix)) + (j - len(snames) / 2 + 0.5) * w
        ax[2].bar(xs, [q[0] for q in v], w, color=c, alpha=0.9 if not h else 0.45, hatch=h, lw=0)
        ax[2].errorbar(xs, [q[0] for q in v], yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set(xticks=range(len(pix)), xticklabels=pix, xlabel=f"one slot's 49 tokens ablated (k = {kmax}); the other slot is occupied")
    ax[2].set_ylabel("Δ log p(target noun) vs clean (scene 95% CI)")
    lo, hi = ax[2].get_ylim()
    ax[2].set_ylim(lo, hi + 0.5 * (hi - lo))
    for j, (key, lab, c, h) in enumerate(snames):
        ax[2].text(0.02, 0.97 - 0.06 * j, lab, color=c, fontsize=7.5, transform=ax[2].transAxes, va="top", alpha=1 if not h else 0.7)
    r_ = {m: rel[(m, kmax)][0] for m in RENDS}
    pt, it, pi = pairs[("printed", "text")], pairs[("image", "text")], pairs[("printed", "image")]
    title = (f"T1 ({name}, {n} scenes, k = {kmax}, {'mean' if mean_used else 'zero'}-ablation). Relative cost of the J-lens part: image {r_['image'][0]:.2f} [{r_['image'][1]:.2f}, {r_['image'][2]:.2f}], "
             f"printed {r_['printed'][0]:.2f} [{r_['printed'][1]:.2f}, {r_['printed'][2]:.2f}], text {r_['text'][0]:.2f} [{r_['text'][1]:.2f}, {r_['text'][2]:.2f}]. "
             f"printed − text {pt[0]:+.2f} [{pt[1]:.2f}, {pt[2]:.2f}]; image − text {it[0]:+.2f} [{it[1]:.2f}, {it[2]:.2f}]; printed − image {pi[0]:+.2f} [{pi[1]:.2f}, {pi[2]:.2f}]. "
             f"Right: the J part's cost is confined to the queried slot.")
    fig.suptitle("\n".join(textwrap.wrap(title, 190)), fontsize=9.5, y=0.985, va="top")
    fig.text(0.05, 0.02, "E2 scenes (COCO crops in six numbered slots); printed = the object's name printed in the slot; the J-lens part = top-256 right singular directions of W_U J_L at the read layers; judge gemini-2.5-flash", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
