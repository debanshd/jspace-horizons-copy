#!/usr/bin/env python
"""Saying against using, by how much of the state is moved. From F4's rank sweep (exploratory): x = the share of scenes on which the receiver
names the donor's object, y = the share on which it applies the animal rule to it, one point per rank for the lens part (solid) and a
random part (dashed), plus the whole state. The diagonal is "uses what it says".
    venv/bin/python jbound/fig_rank.py --run runs/run4/Qwen3.5-4B --out runs/run4/figRank_Qwen3.5-4B.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.load(open(f"{a.run}/eF4_results.json"))
    name = os.path.basename(a.run.rstrip("/"))
    R = r["rank"]
    sw = r.get("rank_sweep", {})
    c = r["carried"]
    pts = {"lens": {}, "random": {}}
    for k, v in sw.items():
        fam = "lens" if k.startswith("lens") else "random"
        rk = int(k.split("r=")[1].split(",")[0])
        pts[fam][rk] = (v["say"], v["animal"])
    pts["lens"][R] = (c["say"]["lens part, read layers"], c["animal"]["lens part, read layers"])
    draws = [k for k in c["say"] if k.startswith("random part, read layers, draw")]
    pts["random"][R] = (tuple(float(np.mean([c["say"][k][i] for k in draws])) for i in range(3)), tuple(float(np.mean([c["animal"][k][i] for k in draws])) for i in range(3)))
    full = (c["say"]["full state, read layers"], c["animal"]["full state, read layers"])
    figstyle.apply(titlesize=10, labelsize=9)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.8)
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.8, ls=":")
    for fam, col, ls, lab, j in (("lens", OKABE[0], "-", "the lens part, by rank", 0), ("random", OKABE[4], "--", "a random part of the same rank", 1)):
        rks = sorted(pts[fam])
        xs, ys = [pts[fam][k][0][0] for k in rks], [pts[fam][k][1][0] for k in rks]
        ax.plot(xs, ys, ls=ls, color=col, lw=2.0, marker="o", ms=5, zorder=3, solid_capstyle="round")
        ax.errorbar(xs, ys, xerr=[[pts[fam][k][0][0] - pts[fam][k][0][1] for k in rks], [pts[fam][k][0][2] - pts[fam][k][0][0] for k in rks]],
                    yerr=[[pts[fam][k][1][0] - pts[fam][k][1][1] for k in rks], [pts[fam][k][1][2] - pts[fam][k][1][0] for k in rks]], fmt="none", ecolor=col, elinewidth=0.6, alpha=0.5)
        groups = {}
        for k, x, y in zip(rks, xs, ys):
            groups.setdefault((round(x, 2), round(y, 2)), []).append(k)
        for (x, y), ks in groups.items():
            ax.annotate("r = " + ", ".join(str(k) for k in ks), (x, y), xytext=(7, 7) if fam == "lens" else (7, -12), textcoords="offset points", fontsize=8, color=col)
        ax.text(0.02, 0.97 - 0.06 * j, lab, color=col, fontsize=8.5, transform=ax.transAxes, va="top", ha="left")
    ax.plot([full[0][0]], [full[1][0]], marker="s", ms=7, color="0.2", lw=0, zorder=4)
    ax.annotate("the whole state", (full[0][0], full[1][0]), xytext=(-8, 8), textcoords="offset points", fontsize=8, color="0.2", ha="right")
    ax.text(0.02, 0.85, "dotted: uses what it says", color="0.5", fontsize=8, transform=ax.transAxes, va="top", ha="left")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="names the donor's object (share of scenes)", ylabel="applies the animal rule to the donor's object (share of scenes)")
    ax.set_title("moved at the middle layers into a receiver that sees its own object", fontsize=9.5)
    l1024 = pts["lens"].get(1024)
    r1024 = pts["random"].get(1024)
    title = (f"{name}: a 256-direction part names and does not use; at 1024 directions the lens-ordered part carries the fact ({l1024[1][0]:.2f}) and a random part of the same size still does not ({r1024[1][0]:.2f})"
             if l1024 and r1024 else f"{name}: saying against using by rank")
    fig.suptitle("\n".join(textwrap.wrap(title, 90)), fontsize=9.5, y=0.98)
    fig.text(0.005, 0.005, f"F4 rank sweep, exploratory, no rule · {r['n']} scenes · 95% scene bootstrap · {name}", fontsize=6.5, color="0.5")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
