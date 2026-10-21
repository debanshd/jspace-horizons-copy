#!/usr/bin/env python
"""Cross-model N1 figure: the causal necessity of the J-readable image-token component by question complexity and model size.
    venv/bin/python jbound/fig_n1.py --runs runs/n1confirm2/Qwen3.5-2B runs/n1confirm2/Qwen3.5-4B runs/n1confirm2/Qwen3.6-27B --out runs/n1confirm2/figN1_models.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle
OKABE = figstyle.OKABE                                                                   # the suite's palette: J-lens 0, keep-only 1, J++ 2, PCA 3, random 4
ARMS = [("ablate J part r=256", "J-lens, r = 256", OKABE[0], "-"), ("ablate J part r=512", "J-lens, r = 512 (d/10)", OKABE[0], "-."),
        ("ablate J++ part r=256", "J++, r = 256", OKABE[2], "-"), ("ablate J++ part r=512", "J++, r = 512", OKABE[2], "-."),
        ("ablate PCA part r=256", "top-256 PCs", OKABE[3], "--"), ("ablate random part r=256", "random, r = 256", OKABE[4], ":"),
        ("ablate random part r=512", "random, r = 512", OKABE[4], "-.")]
BINS = ["2", "3", "4+"]


def spread(items, gap, lo, hi):
    """items = [(y, payload)]: nudge the y's apart by at least gap, keep their order, keep the stack inside [lo, hi]"""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys and ys[-1] > hi:
        ys = [y - (ys[-1] - hi) for y in ys]
    if ys and ys[0] < lo:
        ys = [y + (lo - ys[0]) for y in ys]
    return [(y, t[1]) for y, t in zip(ys, items)]


def jr(r, rank):
    """J minus random at a rank, aggregate over bins: the saved paired interval for 256, point estimates otherwise"""
    if rank == 256 and "J_minus_random_r256" in r:
        return r["J_minus_random_r256"]
    a = r["by_bin"]["all"]
    j, rnd = a.get(f"ablate J part r={rank}"), a.get(f"ablate random part r={rank}")
    return None if j is None or rnd is None else (j["dlogp"][0] - rnd["dlogp"][0], np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [(os.path.basename(r.rstrip("/")), json.load(open(f"{r}/eN1_results.json"))) for r in a.runs]
    figstyle.apply(titlesize=8, labelsize=8.5)
    n = len(runs)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.6 * (n + 1), 5.4), gridspec_kw=dict(wspace=0.36))
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.06, right=0.99)
    x = np.arange(len(BINS))
    lo_all, hi_all, ends_all = 0.0, 0.0, []
    for ax, (name, r) in zip(axes, runs):
        ends = []
        for key, lab, c, ls in ARMS:
            if key not in r["by_bin"]["2"]:
                continue
            m = [r["by_bin"][b][key]["dlogp"][0] for b in BINS]
            lo = [r["by_bin"][b][key]["dlogp"][1] for b in BINS]
            hi = [r["by_bin"][b][key]["dlogp"][2] for b in BINS]
            lo_all, hi_all = min(lo_all, min(lo)), max(hi_all, max(hi))
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(x, m, ls, color=c, lw=1.8, marker="o", ms=3.5)
            ends.append((m[-1], (lab, c)))
        ends_all.append(ends)
        ax.axhline(0, color="0.6", lw=0.8)
        ev = r.get("evidence_dlogp_clean_minus_noimage", [np.nan] * 3)
        en = r.get("energy_removed", {})
        jr256, jr512 = jr(r, 256), jr(r, 512)
        line2 = f"evidence {ev[0]:.1f} nats · J − random {jr256[0]:+.2f} at r = 256" + (f", {jr512[0]:+.2f} at r = 512" if jr512 else "")
        line3 = f"energy removed: J {en.get('J r=256', np.nan):.0%}" + (f" / {en.get('J r=512'):.0%}" if "J r=512" in en else "") + f", PCs {en.get('PCA r=256', np.nan):.0%}, random {en.get('random r=256', np.nan):.0%}"
        ax.set_title(f"{name}\n{line2}\n{line3}", loc="left", pad=8)
        ax.set(xticks=x, xticklabels=[f"{b} ops" for b in BINS], xlim=(-0.3, len(BINS) - 1 + 1.5))
        ax.set_xlabel(f"GQA question program length\n({r['n_images']} images × 1 question per bin)")
    ylo, yhi = lo_all - 0.15, hi_all + 0.15
    for ax, ends in zip(axes[:n], ends_all):
        ax.set_ylim(ylo, yhi)
        for y, (lab, c) in spread(ends, (yhi - ylo) * 0.055, ylo + 0.1, yhi - 0.1):
            ax.text(x[-1] + 0.12, y, lab, color=c, fontsize=7.2, va="center", ha="left")
    axes[0].set_ylabel("Δ log p(reference answer) after ablating the part\nat the read layers of every image token (95% image bootstrap)")
    ax = axes[-1]
    ends = []
    for j, (name, r) in enumerate(runs):
        acc = [r["by_bin"][b]["clean"]["accuracy"] for b in BINS]
        noi = [r["by_bin"][b]["no image"]["accuracy"][0] for b in BINS]
        xs = x + (j - n / 2 + 0.5) * 0.2
        ax.errorbar(xs, [t[0] for t in acc], yerr=[[t[0] - t[1] for t in acc], [t[2] - t[0] for t in acc]], fmt="o", color=OKABE[j], ms=4, capsize=2, lw=1)
        ax.plot(xs, noi, "_", color=OKABE[j], ms=11, mew=1.6)
        ends.append((acc[-1][0], (name, OKABE[j])))
    ax.set(xticks=x, xticklabels=[f"{b} ops" for b in BINS], ylim=(0, 1.0), xlim=(-0.5, len(BINS) - 1 + 1.6), xlabel="GQA question program length")
    for y, (name, c) in spread(ends, 0.06, 0.05, 0.95):
        ax.text(x[-1] + 0.34, y, name, color=c, fontsize=7.2, va="center")
    ax.set_ylabel("accuracy, Gemini judge vs reference\n(dots: with the image; dashes: without)")
    ax.set_title("behaviour\n(the same questions)", loc="left", pad=8)
    parts = []
    for name, r in runs:
        v = jr(r, 256)
        parts.append(f"{name} {v[0]:+.1f}" + (f" [{v[1]:.1f}, {v[2]:.1f}]" if not np.isnan(v[1]) else ""))
    t1 = "N1. Ablating the J-lens-readable part of every image token at the read layers. J − random (r = 256): " + "; ".join(parts) + " nats."
    t2 = "The cost grows with program length on the 2B and 4B; a random part of the same rank costs nothing; on the 27B the top principal components, not the lens-readable part, carry the picture's contribution."
    fig.suptitle("\n".join(textwrap.wrap(t1, 210) + textwrap.wrap(t2, 210)), fontsize=9.5, y=0.985, va="top")
    fig.text(0.06, 0.02, "GQA testdev-balanced, 48 images disjoint from the pilot, one question per program-length bin; the J++ lens where the hub has one; judge gemini-2.5-flash", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
