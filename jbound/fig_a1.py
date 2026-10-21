#!/usr/bin/env python
"""A1 / A2 figure from a run folder: horizontal paired bars per arm (names the object / reads the label), the clean readout, and the
source report. Works for the canvas run (eA1_results.json) and the SCAM run (eA2_results.json, with the injection dose curve).
    venv/bin/python jbound/fig_a1.py --run runs/run1/Qwen3.5-4B --cell A1 --out runs/run1/figA1_canvas.png
"""
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
from figstyle import OKABE

SHORT = {"clean": "no change", "none": "no change", "ablate label word Y": "label word's direction removed", "ablate note word Y": "note word's direction removed",
         "ablate object word X": "object word's direction removed", "ablate other word Z": "another word's direction removed", "ablate unrelated word": "unrelated word's direction removed",
         "ablate random direction": "random direction removed", "ablate note word Y, W_U only": "note word, output-layer direction only",
         "mean-ablate label word Y": "label word's direction, mean-replaced", "mean-ablate object word X": "object word's direction, mean-replaced",
         "ablate J part on the label band": "lens part removed on the label's tokens", "ablate random part on the label band": "random part removed on the label's tokens",
         "ablate J part on the object rows": "lens part removed on the object's tokens", "ablate random part on the object rows": "random part removed on the object's tokens"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cell", default="A1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.load(open(f"{a.run}/e{a.cell}_results.json"))
    name = os.path.basename(a.run.rstrip("/"))
    figstyle.apply(titlesize=9, labelsize=8.5)
    has_inj = "injection" in r
    fig, ax = plt.subplots(1, 3 + has_inj, figsize=(6.2 + 4.4 * (2 + has_inj), 5.2), gridspec_kw=dict(wspace=0.75, width_ratios=[1.9, 1.3] + [1] * (1 + has_inj)))
    fig.subplots_adjust(bottom=0.16, top=0.82, left=0.17, right=0.98)
    arms = list(r["arms"])
    y = np.arange(len(arms))[::-1]
    for j, (col, lab, c) in enumerate((("says_X", "names the object", OKABE[0]), ("says_Y", "reads the written word", OKABE[1]))):
        v = [r["arms"][n][col] for n in arms]
        ax[0].barh(y + (0.5 - j) * 0.38, [t[0] for t in v], 0.36, color=c, alpha=0.9)
        ax[0].errorbar([t[0] for t in v], y + (0.5 - j) * 0.38, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
        ax[0].text(0.02 + 0.5 * j, 1.02, lab, color=c, fontsize=8, transform=ax[0].transAxes, ha="left", va="bottom")
    ax[0].set(yticks=y, yticklabels=[SHORT.get(n, n) for n in arms], xlim=(0, 1.0), xlabel="share of answers (judge), 95% CI over items")
    ax[0].tick_params(axis="y", labelsize=7.5)
    ax[0].axvline(r["arms"][arms[0]]["says_X"][0], color=OKABE[0], lw=0.8, ls=":")
    k = 1
    keys = list(r["readout"])
    v = [r["readout"][kk] for kk in keys]
    cols = [OKABE[5] if "logit" in kk else (OKABE[0] if kk.startswith("X") else OKABE[1]) for kk in keys]
    lab5 = {"X on image tokens": "object, on its image tokens", "Y on image tokens": "word, on its image tokens", "X at the decision position": "object, at the answer position", "Y at the decision position": "word, at the answer position", "X on image tokens (logit lens)": "object, image tokens, logit lens"}
    yy = np.arange(len(keys))[::-1]
    ax[k].barh(yy, [t[0] for t in v], 0.6, color=cols)
    ax[k].errorbar([t[0] for t in v], yy, xerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="none", ecolor="0.3", lw=0.8, capsize=2)
    ax[k].set(yticks=yy, yticklabels=[lab5.get(kk, kk) for kk in keys], xlim=(0, 1.05), xlabel="readable on the unchanged input\n(lens top-10, any read layer)")
    ax[k].tick_params(axis="y", labelsize=7.5)
    k += 1
    if has_inj:
        alphas = [float(x) for x in r["injection"]["note word Y"]]
        for fam, c, ls in (("note word Y", OKABE[1], "-"), ("random, matched norm", OKABE[4], ":")):
            v = [r["injection"][fam][str(x) if str(x) in r["injection"][fam] else str(int(x))] for x in alphas]
            ax[k].errorbar(alphas, [t[0] for t in v], yerr=[[t[0] - t[1] for t in v], [t[2] - t[0] for t in v]], fmt="-o", ls=ls, color=c, ms=4, lw=1.6, capsize=3)
            ax[k].text(alphas[-1] * 1.1, v[-1][0], "the note word's\ndirection" if fam == "note word Y" else "random direction,\nsame norm", color=c, fontsize=7.5, va="center")
        ax[k].axhline(r["injection"]["baseline_says_Y"][0], color="0.5", lw=0.8, ls="--")
        ax[k].set(xscale="log", xticks=alphas, xticklabels=[str(x) for x in alphas], xlim=(min(alphas) * 0.7, max(alphas) * 2.2), ylim=(0, 1.05), xlabel="dose (units of the direction's mean projection)", ylabel="share saying the note word (judge)\non the twin photo, no note")
        ax[k].minorticks_off()
        k += 1
    sr = r["self_report"] if a.cell == "A1" else r["self_report"]
    if a.cell == "A1":
        vals, nn = [r["source_report"]["word_when_following_label"], r["source_report"]["word_when_naming_object"]], [r["source_report"]["n_following"], r["source_report"]["n_naming"]]
        ylab = "share saying the answer came from 'the printed word'"
    else:
        vals, nn = [sr["source_writing_when_following"], sr["source_writing_when_naming_object"]], [sr["n_following"], sr["n_naming"]]
        ylab = "share saying the answer came from 'writing'"
    ax[k].bar([0, 1], vals, 0.6, color=[OKABE[1], OKABE[0]])
    ax[k].set(xticks=[0, 1], xticklabels=[f"answer followed\nthe word (n = {nn[0]})", f"answer named\nthe object (n = {nn[1]})"], ylim=(0, 1.05), ylabel=ylab, xlabel="source report, asked after its own answer")
    ax[k].tick_params(axis="x", labelsize=7.5)
    lf = r["arms"][arms[0]]["says_Y"][0]
    resc = r["rescue_Y"]
    ctrl = r.get("ctrl_Z", r.get("ctrl_unrelated"))
    if has_inj:
        g = r["induction_gap_top_alpha"]
        top = max(float(x) for x in r["injection"]["note word Y"])
        title = (f"{a.cell} ({name}, n = {r['n']}). On real photos the hand-written note wins on only {lf:.2f} of items, although the note word is readable on the image tokens ({r['readout'][keys[1]][0]:.2f}). "
                 f"Removing the object word's direction raises note-following by {r['loss_X'][0]:+.2f} [{r['loss_X'][1]:.2f}, {r['loss_X'][2]:.2f}]. Adding the note word's direction to the note-free twin makes the model say the note word on {r['injection']['note word Y'][str(top)][0]:.2f} of items at dose {top:g} "
                 f"(gap to a random direction of the same norm {g[0]:+.2f} [{g[1]:.2f}, {g[2]:.2f}]).")
    else:
        title = (f"{a.cell} ({name}, n = {r['n']}). The written word wins on {lf:.2f} of items while the object stays readable on its tokens ({r['readout'][keys[0]][0]:.2f}). "
                 f"Removing the word's single lens direction returns the model to the object on {resc[0]:+.2f} [{resc[1]:.2f}, {resc[2]:.2f}] more items; another word's direction {ctrl[0]:+.2f}, a random direction {r['ctrl_random'][0]:+.2f}.")
    fig.suptitle("\n".join(textwrap.wrap(title, 150)), fontsize=9.5, y=0.985, va="top")
    figstyle.finish(fig, a.out)


if __name__ == "__main__":
    main()
