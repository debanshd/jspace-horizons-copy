"""Figures for the read battery (R1), the attention knockout (R2) and the dose-response, drawn from the run CSVs.

    uv run scripts/fig_battery.py --runs runs --models Qwen3.5-4B,Qwen3.5-2B,Qwen3.5-0.8B,gemma-4-E4B-it

Per model: figR1_depth_<model>.png (log-probability curves by patch layer, and the read depth of every question),
figR2_readers_<model>.png (knockout cost by layer against the patching depth), figDose_<model>.png (the mixed donor).
Across models: figR1_models.png (read depth as a share of depth, every question, every model). Fact orange, name blue.
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from read_battery import QUESTIONS  # noqa: E402

BLUE, ORANGE, GREEN, PINK, YELLOW, SKY, BLACK = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"
GREY = "#8a8a8a"


def read_csv(path):
    """a run's CSV, stored as .csv or, in results/, as .csv.gz"""
    p = Path(path)
    return pd.read_csv(p if p.exists() else p.with_name(p.name + ".gz"))


COPIES = {"say", "plural", "french", "letter", "last"}  # answers that copy the word form; every other question is a decision
ATTN = {"Qwen3.5-4B": [3, 7, 11, 15, 19, 23, 27, 31], "Qwen3.5-2B": [3, 7, 11, 15, 19, 23], "Qwen3.5-0.8B": [3, 7, 11, 15, 19, 23]}
LABEL = {
    "say": "say the name",
    "plural": "plural of the name",
    "french": "name in French",
    "letter": "first letter",
    "last": "last letter",
    "nletters": "number of letters",
    "long": "name > 4 letters? (7/3)",
    "before_d": "first letter before D? (7/3)",
    "set1": "is it a {name}? (7/3)",
    "set": "one of 3 names? (7/3)",
    "set5": "one of 5 names? (7/3)",
    "animal": "animal? (7/3)",
    "notanimal": "not an animal? (7/3)",
    "ride": "ridden or driven? (7/3)",
    "legs": "how many legs?",
    "andrule": "animal and > 4 letters? (7/3)",
    "category": "animal, vehicle or other?",
    "yesno": "animal? (yes/no)",
    "mc": "which of 16? (letter)",
}


def style():
    for f in glob.glob("/usr/share/**/*[Hh]eros*.[ot]tf", recursive=True):
        fm.fontManager.addfont(f)
    font = next((f for f in ["TeX Gyre Heros", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in {x.name for x in fm.fontManager.ttflist}), "DejaVu Sans")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": [font],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "axes.titlelocation": "left",
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )


def finish(fig, out):
    sns.despine(fig=fig)
    fig.savefig(out)
    fig.savefig(str(out).replace(".png", ".pdf"))
    fig.savefig(str(out).replace(".png", ".svg"))
    plt.close(fig)
    print("wrote", out)


def boot_depth(d, sweep, n=1000, seed=0):
    """read depth (last layer with mean ld >= 0) with a scene bootstrap: (depth, lo, hi)"""
    scenes = np.sort(d.scene.unique())
    M = d.pivot_table(index="scene", columns="layer", values="ld").reindex(scenes)[sweep].values  # [scene, layer]
    rng = np.random.default_rng(seed)

    def depth(m):
        ok = np.where(m >= 0)[0]
        return sweep[ok.max()] if len(ok) else sweep[0] - 1

    point = depth(M.mean(0))
    bs = [depth(M[rng.integers(len(scenes), size=len(scenes))].mean(0)) for _ in range(n)]
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(runs, model):
    R = read_csv(runs / "run7" / model / "read_battery.csv")
    S = json.load(open(runs / "run7" / model / "summary.json"))
    return R, S


def fig_depth(runs, model, out):
    R, S = load(runs, model)
    sweep, n_layers = S["sweep"], S["n_layers"]
    P = R[(R.kind == "patch") & R.distinct]
    tasks = [t for t in QUESTIONS if t in S["questions"] and S["questions"][t]["gate_pass"]]
    dropped = [t for t in QUESTIONS if t in S["questions"] and not S["questions"][t]["gate_pass"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    show = [("say", BLUE, "say the name"), ("letter", SKY, "first letter"), ("category", GREEN, "animal, vehicle or other?"), ("set5", PINK, "one of 5 names?"), ("animal", ORANGE, "animal? (7/3)")]
    for t, c, lab in show:
        if t not in tasks:
            continue
        d = P[P.task == t]
        g = d.groupby("layer").ld
        m, lo, hi = g.mean().reindex(sweep), g.quantile(0.025).reindex(sweep), g.quantile(0.975).reindex(sweep)
        a.plot(sweep, m.values, color=c, lw=2)
        a.fill_between(sweep, lo.values, hi.values, color=c, alpha=0.1, lw=0)
        a.text(sweep[-1] + 0.4, m.values[-1] if t != "say" else m.values[-1] + 0.6, lab, color=c, fontsize=8, va="center")
    a.axhline(0, color=BLACK, lw=0.6)
    for L in ATTN.get(model, []):
        a.axvline(L, color=GREY, lw=0.5, ls=":")
    a.set_xlabel("layer at which the donor's image-token state is written in")
    a.set_ylabel("log p(X's answer) - log p(Z's answer), nats")
    a.set_title(f"{model}: the donor's object wins until the question's read depth,\nthen loses within a few layers")
    a.set_xlim(-1.5, n_layers + 0.5)
    rows = []
    for t in tasks:
        dpt, lo, hi = boot_depth(P[P.task == t], sweep)
        rows.append((t, dpt, lo, hi, QUESTIONS[t][2], QUESTIONS[t][1], S["questions"][t]["n"]))
    rows.sort(key=lambda r: (r[1], r[0]))
    for i, (t, dpt, lo, hi, _kind, _needs, n) in enumerate(rows):
        c = BLUE if t in COPIES else ORANGE
        b.plot([lo, hi], [i, i], color=c, lw=1.2, alpha=0.6)
        b.plot(dpt, i, "o", color=c, ms=6)
        b.text(-2.2, i, LABEL[t], ha="right", va="center", fontsize=7.5, color=c)
        b.text(n_layers + 0.3, i, f"n {n}", ha="left", va="center", fontsize=6.5, color=GREY)
    for L in ATTN.get(model, []):
        b.axvline(L, color=GREY, lw=0.5, ls=":")
    b.set_yticks([])
    b.set_xlim(-1.5, n_layers + 0.5)
    b.set_ylim(-0.8, len(rows) - 0.2)
    b.set_xlabel("read depth: last layer at which X's answer is still the more likely")
    b.set_title("every decision reads early (orange), every copy of the word form reads late (blue),\nwhatever the question needs and whatever form the answer takes" if rows else "")
    foot = f"n = {S['n']} scenes (scenes where X's and Z's answers coincide are excluded per question); bands and bars are 95% scene bootstraps; dotted lines mark standard-attention layers"
    if dropped:
        foot += f"; dropped by the clean gate (< 0.85): {', '.join(dropped)}"
    fig.text(0.01, -0.02, foot, fontsize=7, color=GREY)
    fig.tight_layout()
    finish(fig, out)
    return rows


def fig_readers(runs, model, out, depths):
    K = read_csv(runs / "run7k" / model / "attention_knockout.csv")
    KS = json.load(open(runs / "run7k" / model / "summary.json"))
    AL = KS["attention_layers"]
    K = K[K.distinct]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (t, c) in zip(axes, (("animal", ORANGE), ("say", BLUE))):
        d = K[K.task == t]
        clean = d[d.kind == "clean"].set_index("scene").ld
        for kind, ls, lab in (("from", "-", "blocked from layer L on"), ("upto", "--", "blocked up to layer L")):
            cost = []
            for L in AL:
                g = d[(d.kind == kind) & (d.layer == L)].set_index("scene").ld
                cost.append((clean - g.reindex(clean.index)).mean())
            ax.plot(AL, cost, ls=ls, color=c, lw=2, marker="o", ms=4)
            ax.text(AL[-1] + 0.3, cost[-1], lab, color=c, fontsize=7.5, va="center")
        single = [(clean - d[(d.kind == "single") & (d.layer == L)].set_index("scene").ld.reindex(clean.index)).mean() for L in AL]
        ax.bar(AL, single, width=1.6, color=c, alpha=0.3, lw=0)
        ax.text(AL[0], max(single) + 0.3, "bars: one layer blocked", color=c, fontsize=7.5)
        if t in depths:
            ax.axvline(depths[t] + 0.5, color=BLACK, lw=1, ls=":")
            ax.text(depths[t] + 0.8, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, f"patching depth {depths[t]}", fontsize=7.5, va="top")
        ax.axhline(0, color=BLACK, lw=0.6)
        ax.set_xlabel("standard-attention layer L")
        ax.set_ylabel("cost of the knockout, nats")
        ax.set_title(f"{LABEL[t]}: readers {KS['questions'][t]['readers']}, last reader {KS['questions'][t]['last_reader']}")
    fig.text(
        0.01,
        -0.03,
        f"{model}; attention from the question to the image tokens blocked at the listed layers; n = {KS['n']} scenes; cost = clean log-probability difference minus the knockout's",
        fontsize=7,
        color=GREY,
    )
    fig.tight_layout()
    finish(fig, out)


def fig_dose(runs, model, out):
    R, S = load(runs, model)
    sweep = S["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (t, c) in zip(axes, (("animal", ORANGE), ("say", BLUE))):
        d = R[(R.task == t) & R.distinct]
        for kind, a, alpha, lab in (
            ("mix", 0.25, 0.35, "donor weight 0.25"),
            ("mix", 0.5, 0.55, "0.5"),
            ("mix", 0.75, 0.8, "0.75"),
            ("patch", 1.0, 1.0, "1 (the full state)"),
            ("mix norm", 0.5, 0.55, "0.5, norm-matched"),
        ):
            g = d[(d.kind == kind) & (d.alpha == a)].groupby("layer").ld.mean().reindex(sweep)
            ax.plot(sweep, g.values, color=c if kind != "mix norm" else GREY, lw=2 if kind == "patch" else 1.4, alpha=alpha, ls="-" if kind != "mix norm" else "--")
            ax.text(sweep[0] - 0.3, g.values[0], lab, ha="right", va="center", fontsize=7, color=c if kind != "mix norm" else GREY)
        ax.axhline(0, color=BLACK, lw=0.6)
        ax.set_xlabel("layer at which the mixed state is written in")
        ax.set_ylabel("log p(X's answer) - log p(Z's answer), nats")
        dd = S["dose"].get(t, {})
        ax.set_title(f"{LABEL[t]}: depth by dose " + ", ".join(f"{k.split()[-1]}: {v['depth_ld']}" for k, v in dd.items()))
        ax.set_xlim(sweep[0] - 6, sweep[-1] + 1)
    fig.text(0.01, -0.03, f"{model}; x <- (1 - a) receiver + a donor at one layer; n = {S['n']} scenes", fontsize=7, color=GREY)
    fig.tight_layout()
    finish(fig, out)


def fig_models(runs, models, out, all_rows):
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.6 * len(models)))
    order = [t for t in QUESTIONS]
    for j, model in enumerate(models):
        S = json.load(open(runs / "run7" / model / "summary.json"))
        n_layers = S["n_layers"]
        for t, dpt, lo, hi, _kind, _needs, _n in all_rows.get(model, []):
            c = BLUE if t in COPIES else ORANGE
            x = (dpt + 1) / n_layers
            ax.plot([(lo + 1) / n_layers, (hi + 1) / n_layers], [j, j], color=c, lw=1, alpha=0.5)
            ax.plot(x, j, "o", color=c, ms=5, alpha=0.85)
            ax.text(x, j + 0.12 + 0.12 * (order.index(t) % 3), LABEL[t].split(" (")[0], fontsize=5.5, color=c, ha="center", va="bottom", rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("read depth as a share of the model's layers")
    ax.set_title("every decision (orange) stops reading the image early, every copy of the word form (blue) late, in every model")
    fig.text(0.01, -0.05, "point = last layer at which the donor's answer is still the more likely, bar = 95% scene bootstrap; questions that fail the clean gate are omitted", fontsize=7, color=GREY)
    fig.tight_layout()
    finish(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--models", default="Qwen3.5-4B,Qwen3.5-2B,Qwen3.5-0.8B,gemma-4-E4B-it")
    ap.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = ap.parse_args()
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    done = []
    for model in args.models.split(","):
        if not (args.runs / "run7" / model / "summary.json").exists():
            print("no battery summary for", model)
            continue
        rows = fig_depth(args.runs, model, args.out / f"figR1_depth_{model}.png")
        all_rows[model] = rows
        done.append(model)
        depths = {t: r[1] for r in rows for t in [r[0]]}
        if (args.runs / "run7k" / model / "summary.json").exists():
            fig_readers(args.runs, model, args.out / f"figR2_readers_{model}.png", depths)
        S = json.load(open(args.runs / "run7" / model / "summary.json"))
        if S.get("dose"):
            fig_dose(args.runs, model, args.out / f"figDose_{model}.png")
    if done:
        fig_models(args.runs, done, args.out / "figR1_models.png", all_rows)
        more(args.runs, done, args.out)


# ---- consistency across concepts (Noah, 20 Oct) and the one-scene example


def fig_concepts(runs, model, out):
    """per-scene read depth of say and animal (last layer with ld >= 0 per scene), by the donor's object"""
    R, S = load(runs, model)
    sweep = S["sweep"]
    P = R[(R.kind == "patch") & R.distinct]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    stats = {}
    for ax, (t, c) in zip(axes, (("animal", ORANGE), ("say", BLUE))):
        d = P[P.task == t]
        M = d.pivot_table(index="scene", columns="layer", values="ld").reindex(columns=sweep)
        per_scene = M.apply(lambda row: max([L for L, v in zip(sweep, row.values) if v >= 0], default=sweep[0] - 1), axis=1)
        obj = d.groupby("scene").X.first().reindex(per_scene.index)
        df = pd.DataFrame(dict(depth=per_scene.values, X=obj.values))
        order = df.groupby("X").depth.median().sort_values().index.tolist()
        for i, x in enumerate(order):
            v = df[df.X == x].depth.values
            ax.scatter(v + np.random.default_rng(i).uniform(-0.25, 0.25, len(v)), [i] * len(v), s=14, color=c, alpha=0.6, lw=0)
            ax.text(-2.4, i, f"{x} (n {len(v)})", ha="right", va="center", fontsize=7.5)
        iqr = float(np.percentile(df.depth, 75) - np.percentile(df.depth, 25))
        within = float((np.abs(df.depth - df.depth.median()) <= 1).mean())
        stats[t] = dict(median=float(df.depth.median()), iqr=iqr, share_within_1_layer=within, by_object={x: [int(v) for v in df[df.X == x].depth.values] for x in order})
        for L in ATTN.get(model, []):
            ax.axvline(L, color=GREY, lw=0.5, ls=":")
        ax.set_yticks([])
        ax.set_xlim(-1.5, S["n_layers"] + 0.5)
        ax.set_xlabel("read depth of this scene (last layer at which X's answer is the more likely)")
        ax.set_title(f"{LABEL[t]}: median {df.depth.median():.0f}, {within:.0%} of scenes within one layer of it")
    fig.text(0.01, -0.03, f"{model}; one point per scene, grouped by the donor's object; n = {S['n']} scenes", fontsize=7, color=GREY)
    fig.tight_layout()
    finish(fig, out)
    return stats


def fig_example(runs, model, out, scene=0, tasks=("say", "letter", "category", "set", "ride", "animal")):
    """one scene, every question, the model's answer at every patch layer: X's answer in blue, Z's in orange, other in grey"""
    R, S = load(runs, model)
    sweep = S["sweep"]
    d = R[(R.scene == scene) & (R.kind == "patch")]
    tasks = [t for t in tasks if t in set(d.task)]
    X, Z = d.X.iloc[0], d.Z.iloc[0]
    fig, ax = plt.subplots(figsize=(0.32 * len(sweep) + 2.6, 0.42 * len(tasks) + 1.2))
    for i, t in enumerate(tasks):
        g = d[d.task == t].set_index("layer").reindex(sweep)
        for j, L in enumerate(sweep):
            row = g.loc[L]
            col = BLUE if row.follows_X else ORANGE if row.keeps_Z else GREY
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=col, alpha=0.18, lw=0))
            ax.text(j + 0.5, -i + 0.5, str(row.generation)[:7], ha="center", va="center", fontsize=5.5, color=col, rotation=90 if len(str(row.generation)) > 3 else 0)
        ax.text(-0.3, -i + 0.5, LABEL[t], ha="right", va="center", fontsize=7.5)
    ax.set_xlim(0, len(sweep))
    ax.set_ylim(-len(tasks) + 1, 1)
    ax.set_xticks([j + 0.5 for j in range(len(sweep))])
    ax.set_xticklabels([str(L) for L in sweep], fontsize=6)
    ax.set_yticks([])
    ax.set_xlabel("layer at which the donor's image-token states are written in")
    ax.set_title(f"{model}, scene {scene}: donor {X}, receiver {Z}; the answer names {X} (blue) or follows {Z} (orange) at each layer")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    finish(fig, out)


def more(runs, models, out):
    stats = {}
    for model in models:
        if (runs / "run7" / model / "summary.json").exists():
            stats[model] = fig_concepts(runs, model, out / f"figConcepts_{model}.png")
            for sc in (0, 1):
                fig_example(runs, model, out / f"figExample_{model}_scene{sc}.png", scene=sc)
    json.dump(stats, open(out / "concepts.json", "w"), indent=2)
    return stats


if __name__ == "__main__":
    main()


def fig_matrix(runs, model, out, run="run7k", what="the image tokens"):
    """the readers matrix: single-layer knockout cost (nats) for every question (rows, copies above decisions) at every softmax layer"""
    K = json.load(open(runs / run / model / "summary.json"))
    AL = K["attention_layers"]
    Q = K["questions"]
    copies = [t for t in QUESTIONS if t in Q and t in COPIES]  # copies of the word form; every other question is a decision
    decisions = [t for t in QUESTIONS if t in Q and t not in copies]
    rows = copies + decisions
    M = np.array([[Q[t]["single"][str(L)]["cost"][0] for L in AL] for t in rows])
    passed = [Q[t]["clean_correct"] >= 0.85 for t in rows]
    fig, ax = plt.subplots(figsize=(min(11.5, 0.55 * len(AL) + 4.2), 0.3 * len(rows) + 1.6))  # a full-layer sweep (Gemma, 42 columns) stays page-wide
    vmax = max(1.0, float(np.nanmax(M)))
    for i, t in enumerate(rows):
        for j, _L in enumerate(AL):
            v = M[i, j]
            c = BLUE if t in copies else ORANGE
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=c, alpha=min(0.95, max(0.0, v / vmax)), lw=0))
            if v >= 0.5:
                ax.text(j + 0.5, -i + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v / vmax > 0.55 else BLACK)
        ax.text(-0.15, -i + 0.5, LABEL[t] + ("" if passed[i] else " (gate failed)"), ha="right", va="center", fontsize=7.5, color=(BLUE if t in copies else ORANGE) if passed[i] else GREY)
    ax.axhline(-len(copies) + 1, color=BLACK, lw=0.8)
    ax.set_xlim(0, len(AL))
    ax.set_ylim(-len(rows) + 1, 1)
    ax.set_xticks([j + 0.5 for j in range(len(AL))])
    ax.set_xticklabels([str(L) for L in AL])
    ax.set_yticks([])
    ax.set_xlabel(f"softmax-attention layer whose reads of {what} are blocked")
    ax.set_title(f"{model}: cost in nats of blocking one layer's reads; copies of the word form (blue) and decisions (orange)")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(
        0.01, -0.02, f"n = {K['n']} scenes; cost = clean log p(correct) - log p(the other kind's answer), minus the same under the knockout; numbers shown when >= 0.5 nats", fontsize=7, color=GREY
    )
    fig.tight_layout()
    finish(fig, out)


def fig_heads(runs, model, out, layers, questions=("animal", "set", "category", "say")):
    """head-level knockout: the cost of blocking one head's reads, per head (columns) at each station layer (rows), per question"""
    K = json.load(open(runs / "run7h" / model / "summary.json"))
    Q = K["questions"]
    H = max(int(h) for h in Q[questions[0]][f"head {layers[0]}"])
    fig, axes = plt.subplots(1, len(questions), figsize=(2.9 * len(questions) + 1, 0.55 * len(layers) + 1.6), sharey=True)
    vmax = max(Q[t][f"head {L}"][str(h)]["cost"][0] for t in questions for L in layers for h in range(H + 1))
    for ax, t in zip(axes, questions):
        c = BLUE if t in COPIES else ORANGE
        for i, L in enumerate(layers):
            lc = Q[t]["single"][str(L)]["cost"][0]
            for h in range(H + 1):
                v = Q[t][f"head {L}"][str(h)]["cost"][0]
                ax.add_patch(plt.Rectangle((h, -i), 1, 1, facecolor=c, alpha=min(0.95, max(0.0, v / vmax)), lw=0))
                if v >= 0.5:
                    ax.text(h + 0.5, -i + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v / vmax > 0.55 else BLACK)
            ax.text(H + 1.15, -i + 0.5, f"layer {lc:.1f}", ha="left", va="center", fontsize=6.5, color=GREY)
        ax.set_xlim(0, H + 1)
        ax.set_ylim(-len(layers) + 1, 1)
        ax.set_xticks([h + 0.5 for h in range(H + 1)])
        ax.set_xticklabels([str(h) for h in range(H + 1)], fontsize=7)
        ax.set_xlabel("query head")
        ax.set_title(LABEL[t], color=c)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks([-i + 0.5 for i in range(len(layers))])
    axes[0].set_yticklabels([f"layer {L}" for L in layers], fontsize=8)
    fig.suptitle(f"{model}: cost in nats of blocking one head's reads of the image tokens (grey: the whole layer blocked)", x=0.01, ha="left", fontsize=9.5)
    fig.text(
        0.01, -0.03, f"n = {K['n']} scenes; attention from the text after the image to the image tokens blocked for one query head at one layer; numbers shown when >= 0.5 nats", fontsize=7, color=GREY
    )
    fig.tight_layout()
    finish(fig, out)


def fig_count(runs, model, out):
    """CountBench: accuracy by the softmax layer from which reads are blocked, one line per count group; and, per count, the share of
    clean accuracy that survives when the image is blocked from layer 15 on (the first softmax layer after the decision station)"""
    R = read_csv(runs / "run10" / model / "bench_knockout.csv")
    AL = sorted(int(L) for L in R[R.kind == "from"].layer.dropna().unique())
    L_after = 15 if 15 in AL else AL[len(AL) // 2]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    groups = [g for g in ("count 2 to 4", "count 5 to 6", "count 7 to 10") if g in set(R.group)]
    cols = [SKY, BLUE, "#08306b"]
    for g, col in zip(groups, cols):
        d = R[R.group == g]
        acc = [d[(d.kind == "from") & (d.layer == L)].correct.mean() for L in AL]
        lo = [np.percentile([d[(d.kind == "from") & (d.layer == L)].correct.sample(frac=1, replace=True, random_state=k).mean() for k in range(200)], 2.5) for L in AL]
        hi = [np.percentile([d[(d.kind == "from") & (d.layer == L)].correct.sample(frac=1, replace=True, random_state=k).mean() for k in range(200)], 97.5) for L in AL]
        a.plot(AL, acc, color=col, lw=2, marker="o", ms=3.5)
        a.fill_between(AL, lo, hi, color=col, alpha=0.12, lw=0)
        a.text(AL[2] - 0.4, acc[2], f"{g.replace('count ', 'counts ')} (n {len(d[d.kind == 'clean'])})", color=col, fontsize=7.5, ha="right", va="center")
    a.axvline(L_after - 0.5, color=GREY, lw=0.6, ls=":")
    a.set_xlim(AL[0] - 1, AL[-1] + 1)
    a.set_ylim(0, 1.03)
    a.set_xlabel("softmax layer L: the question's reads of the image blocked from L on")
    a.set_ylabel("accuracy (items correct on the clean pass)")
    a.set_title(f"{model} on CountBench: the larger the count,\nthe deeper the picture is still read")
    counts = sorted(int(c) for c in R.level.unique())
    pts = []
    for c in counts:
        d = R[R.level == c]
        clean = d[d.kind == "clean"].correct.mean()
        g = d[(d.kind == "from") & (d.layer == L_after)].correct
        bs = [g.sample(frac=1, replace=True, random_state=k).mean() for k in range(500)]
        pts.append((c, g.mean() / clean, np.percentile(bs, 2.5) / clean, np.percentile(bs, 97.5) / clean, len(g)))
    for c, m, lo, hi, n in pts:
        b.plot([c, c], [lo, hi], color=ORANGE, lw=1.2, alpha=0.6)
        b.plot(c, m, "o", color=ORANGE, ms=6)
        b.text(c, hi + 0.03, f"n {n}", ha="center", fontsize=6.5, color=GREY)
    b.set_xticks(counts)
    b.set_ylim(0, 1.15)
    b.set_xlabel("count asked for")
    b.set_ylabel("accuracy as a share of the clean accuracy")
    b.set_title(f"what each count still needs from the picture\nafter the decision station (reads blocked from layer {L_after} on)")
    fig.text(
        0.01,
        -0.03,
        "CountBench (vikhyatk/CountBenchQA); items the model answers correctly on the clean pass; exact match on the first number; bands and bars are 95% item bootstraps",
        fontsize=7,
        color=GREY,
    )
    fig.tight_layout()
    finish(fig, out)
    return pts


def fig_gqa(runs, model, out):
    """GQA photos: judged accuracy per question kind by the layer from which a blank image's token states are written into the photo's tokens"""
    R = read_csv(runs / "run9p" / model / "gqa_knockout.csv")
    Ls = sorted(int(L) for L in R[R.kind == "from"].layer.dropna().unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for k, c, lab in (("decision", ORANGE, "yes / no and either / or"), ("name", BLUE, "what is it (a name)"), ("attribute", SKY, "what colour / material (an attribute)")):
        d = R[R.qkind == k]
        clean = d[d.kind == "clean"].correct.astype(float).mean()
        acc = [d[(d.kind == "from") & (d.layer == L)].correct.astype(float).mean() for L in Ls]
        ax.plot(Ls, acc, color=c, lw=2, marker="o", ms=3)
        ax.axhline(clean, color=c, lw=0.7, ls=":")
        ax.text(Ls[-1] + 0.4, acc[-1], lab, color=c, fontsize=7.5, va="center")
    for L in ATTN.get(model, []):
        ax.axvline(L, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("layer L from which the photo's image tokens carry a blank image's states instead")
    ax.set_ylabel("judged accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{model} on GQA photos: decisions need the picture only through layer 10,\nnames through 18, attributes through 22")
    fig.text(0.01, -0.03, "48 GQA testdev-balanced photos, one question of each kind per photo, scored by the Gemini judge; dotted horizontals are the clean accuracies", fontsize=7, color=GREY)
    fig.tight_layout()
    finish(fig, out)


# ---- the cross-modal interchange (R9): is what a reader head delivers the same object whether the thing was seen or read?


def fig_cross_modal(runs, model, out, layer=19, questions=("animal", "say"), run="run13"):
    """the share of scenes on which the picture receiver's answer follows the typed donor's object, per head of the name station and
    per whole or joint layer (filled), beside the picture donor (hollow). One question: does what the model read drive what it answers
    about what it sees?"""
    S = json.load(open(runs / run / model / "summary.json"))
    layers = [int(L) for L in S["layers"]]
    H = int(S["heads"])
    Q = S["questions"]
    joint = [n for n in Q[questions[0]]["typed>picture"] if n.startswith("all heads of") and "+" in n]
    arms = [f"head {h} of {layer}" for h in range(H)] + [f"all heads of {L}" for L in layers] + joint
    ticks = [f"head {h}" for h in range(H)] + [f"layer {L}" for L in layers] + [n.replace("all heads of ", "layers ").replace("+", ", ") for n in joint]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(0.33 * len(arms) + 1.6, 3.0))
    for k, t in enumerate(questions):
        c = BLUE if t in COPIES else ORANGE
        off = (k - (len(questions) - 1) / 2) * 0.36
        for direction, filled in (("typed>picture", True), ("picture>picture", False)):
            if direction not in Q[t]:
                continue
            e = Q[t][direction]
            m = np.array([e[a]["follows_X"][0] if a in e else np.nan for a in arms])
            lo = np.array([e[a]["follows_X"][1] if a in e else np.nan for a in arms])
            hi = np.array([e[a]["follows_X"][2] if a in e else np.nan for a in arms])
            label = f"{LABEL[t]}, donor {'read the word' if filled else 'saw the picture'}"
            ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="o", ms=4.2, mfc=c if filled else "white", mec=c, ecolor=c, elinewidth=0.8, capsize=0, lw=0, label=label)
            if filled:
                ax.axhline(e["clean"]["follows_X"][0], color=c, lw=0.7, ls=":", alpha=0.7)
    ax.axhline(0.5, color=GREY, lw=0.6, ls="--")
    ax.text(len(arms) - 0.4, 0.515, "rule: 0.5", ha="right", va="bottom", fontsize=6.5, color=GREY)
    ax.axvspan(-0.5, H - 0.5, color=GREY, alpha=0.05, lw=0)
    ax.axvspan(H + len(layers) - 0.5, len(arms) - 0.5, color=GREY, alpha=0.05, lw=0)
    for x0, lab in ((H / 2 - 0.5, f"single heads of layer {layer}"), (H + len(layers) / 2 - 0.5, "whole layers"), (H + len(layers) + len(joint) / 2 - 0.5, "joint layers")):
        ax.text(x0, 1.06, lab, ha="center", va="bottom", fontsize=7, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("share of scenes whose answer\nfollows the donor's object")
    ax.set_title(f"{model}: the donor's head outputs, recorded while it read the word, written into the model that sees another object", pad=14)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.text(
        0.01,
        -0.08,
        f"n = {S['n']} scenes; donor: 'Image slot q contains a X.' with no picture (filled) or the picture of X (hollow); receiver: the picture of Z, the same question;\n"
        "the per-head attention outputs at the question's tokens are written in; dotted: clean; 95% scene bootstraps",
        fontsize=6.5,
        color=GREY,
    )
    fig.tight_layout()
    finish(fig, out)


def fig_cross_align(runs, model, out, question="animal", run="run13"):
    """per head and layer, how much closer a head's output is to the same object typed than to the other object typed (cosine, mean
    over scenes), at the question token where that difference is largest. One question: is the vector the same for seen and read?"""
    S = json.load(open(runs / run / model / "summary.json"))
    layers = [int(L) for L in S["layers"]]
    H = int(S["heads"])
    A = S["alignment"]
    pos_file = runs / (run + "a") / model / "alignment_positions.csv"
    if pos_file.exists():
        P = read_csv(pos_file)
        P = P[P.task == question].groupby(["layer", "head", "pos"]).agg(w=("within", "mean"), a=("across_from_picture", "mean")).reset_index()
        P["d"] = P.w - P.a
        best = P.sort_values("d", ascending=False).groupby(["layer", "head"]).head(1).set_index(["layer", "head"])
        vals = np.array([[best.loc[(L, h), "d"] for h in range(H)] for L in layers])
        where = np.array([[int(best.loc[(L, h), "pos"]) for h in range(H)] for L in layers])
    else:
        vals = np.array([[A[question][str(L)][str(h)]["within_minus_across"][0] for h in range(H)] for L in layers])
        where = None
    fig, ax = plt.subplots(figsize=(0.55 * H + 1.8, 0.5 * len(layers) + 1.6))
    vmax = max(0.05, float(np.nanmax(vals)))
    for i in range(len(layers)):
        for h in range(H):
            v = vals[i, h]
            ax.add_patch(plt.Rectangle((h, -i), 1, 1, facecolor=BLACK, alpha=min(0.9, max(0.0, v / vmax)), lw=0))
            if v >= 0.1:
                ax.text(h + 0.5, -i + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v / vmax > 0.5 else BLACK)
    ax.set_xlim(0, H)
    ax.set_ylim(-len(layers) + 1, 1)
    ax.set_xticks([h + 0.5 for h in range(H)])
    ax.set_xticklabels([str(h) for h in range(H)], fontsize=7)
    ax.set_yticks([-i + 0.5 for i in range(len(layers))])
    ax.set_yticklabels([f"layer {L}" for L in layers], fontsize=8)
    ax.set_xlabel("query head")
    ax.set_title(f"{model}: how much closer a head's output is to the same object read than to the other object read", pad=8)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    note = "at the question token where the difference is largest" if where is not None else "at the answer position"
    foot = f"n = {S['n']} scenes, the {LABEL[question]} question; cosine(X seen, X read) minus cosine(X seen, Z read), mean over scenes,\n{note}; numbers from 0.10"
    fig.text(0.01, -0.1, foot, fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


def fig_cross_example(runs, model, out, scene=0, layer=19, run="run13", questions=("say", "animal")):
    """one scene: the picture the model sees, the sentence the donor read, and the answers clean and with each head of the station written in"""
    R = read_csv(runs / run / model / "cross_modal_patch.csv")
    d = R[(R.scene == scene) & (R.direction == "typed>picture")]
    X, Z = d.X.iloc[0], d.Z.iloc[0]
    arms = ["clean", f"head 5 of {layer}", f"all heads of {layer}", "all heads of 7+11", "all heads of 11+15+19", "all heads of 7+11+15+19"]
    arms = [a for a in arms if a in set(d.arm)]
    fig, ax = plt.subplots(figsize=(0.9 * len(arms) + 2.4, 0.5 * len(questions) + 1.6))
    for i, t in enumerate(questions):
        g = d[d.task == t].set_index("arm").reindex(arms)
        for j, a in enumerate(arms):
            row = g.loc[a]
            col = BLUE if row.follows_X else ORANGE if row.keeps_Z else GREY
            ax.add_patch(plt.Rectangle((j, -i), 1, 1, facecolor=col, alpha=0.18, lw=0))
            ax.text(j + 0.5, -i + 0.5, str(row.generation)[:8], ha="center", va="center", fontsize=6.5, color=col)
        ax.text(-0.3, -i + 0.5, LABEL[t], ha="right", va="center", fontsize=7.5)
    ax.set_xlim(0, len(arms))
    ax.set_ylim(-len(questions) + 1, 1)
    ax.set_xticks([j + 0.5 for j in range(len(arms))])
    ax.set_xticklabels([a.replace("all heads of ", "layers ").replace("+", ", ").replace(" of ", " of layer ") for a in arms], fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("which heads' outputs, recorded while the model read the sentence, are written in")
    ax.set_title(f"{model}, scene {scene}: the model sees a {Z}; the donor read 'Image slot contains a {X}.'; answers that follow the word in blue, the picture in orange")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    finish(fig, out)


# ---- the source report after an interchange (R10, R10b): does the model say it saw what it in fact read?


def fig_source_report(runs, models, out, runs_used=("run15", "run15b")):
    """per written-in layer group, the share of scenes on which the picture receiver names the read donor's object (grey), and among
    those, the share reporting that it saw it (orange) or read it (blue). One question: does the report follow the content's channel?"""
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        rows = {}
        n = None
        for r in runs_used:
            f = runs / r / model / "summary.json"
            if not f.exists():
                continue
            S = json.load(open(f))
            n = S["n"]
            for name, e in S["arms"]["q"].items():
                if name.startswith("all heads of"):
                    rows[name.replace("all heads of ", "").replace("+", ", ")] = e
        order = sorted(rows, key=lambda k: (len(k.split(", ")), k))
        x = np.arange(len(order))
        for i, k in enumerate(order):
            e = rows[k]
            nx = e["names_X"][0]
            ax.bar(i, nx, width=0.62, color=GREY, alpha=0.25, lw=0)
            if e["seen_given_X"]:
                seen, read = e["seen_given_X"][0], e["read_given_X"][0]
                ax.bar(i, nx * seen, width=0.62, color=ORANGE, lw=0)
                ax.bar(i, nx * read, width=0.62, bottom=nx * seen, color=BLUE, lw=0)
                ax.text(i, nx + 0.02, f"seen {seen:.2f}\nread {read:.2f}", ha="center", va="bottom", fontsize=6.2, color=BLACK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"layers {k}" if "," in k else f"layer {k}" for k in order], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{model} (n = {n})")
        ax.axhline(1.0, color=GREY, lw=0.5, ls=":")
    axes[0].set_ylabel("share of scenes naming the read object")
    axes[0].bar([np.nan], [0], color=ORANGE, label="names it and says it saw it")
    axes[0].bar([np.nan], [0], color=BLUE, label="names it and says it read it")
    axes[0].bar([np.nan], [0], color=GREY, alpha=0.25, label="names it, no channel parsed")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("the model sees a cat and has the heads that read 'dog' written in: does it say it saw the dog or read it?", x=0.01, y=1.04, ha="left", fontsize=9.5)
    fig.text(
        0.01,
        -0.1,
        "the receiver sees Z in slot q and reads a note about slot p; the donor read notes about q (X) and p and saw no picture; the donor's per-head attention outputs\n"
        "at the shared suffix are written in at the listed layers; the question asks for the object in slot q and whether it was seen in the picture or read in the note",
        fontsize=6.5,
        color=GREY,
    )
    fig.tight_layout()
    finish(fig, out)


# ---- the whiteboard: the circuit in one drawing


def fig_schematic(out):
    """the reading circuit as found on Qwen3.5-2B: two inputs, three reads, three kinds of answer, with the numbers that carry each arrow"""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color, fc="white", fs=8, lw=1.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=color, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLACK, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color, text="", dy=2.2):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, shrinkA=2, shrinkB=2))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.3, color=color)

    # inputs
    box(2, 62, 22, 22, "a picture of a dog\nin slot 3", GREY, fc="#f3f3f3")
    box(2, 16, 22, 22, "the sentence\n'slot 3 contains a dog'", GREY, fc="#f3f3f3")
    # readers
    box(34, 68, 25, 16, "heads 1 and 3, layer 7\nwhat it is", ORANGE)
    box(34, 40, 25, 16, "layers 11, 15, 19 (head 5)\nthe word form", BLUE)
    box(34, 12, 25, 16, "layers 7 to 15\nwhich channel it came from", GREEN)
    # answers
    box(79, 68, 20, 16, "every decision\nanimal? ridden?\none of 16?", ORANGE, fs=7.5)
    box(79, 40, 20, 16, "the name\nits spelling, French", BLUE, fs=7.5)
    box(79, 12, 20, 16, "'seen' or 'read'", GREEN, fs=7.5)
    # arrows from both inputs to both content readers
    arrow(24, 76, 34, 78, ORANGE)
    arrow(24, 30, 34, 74, ORANGE)
    arrow(24, 70, 34, 50, BLUE)
    arrow(24, 27, 34, 46, BLUE)
    arrow(24, 24, 34, 20, GREEN)
    arrow(24, 66, 34, 23, GREEN)
    arrow(59, 74, 79, 74, ORANGE, "necessary: 2 to 4 nats\ninterchangeable: 0.98", dy=1.5)
    arrow(59, 46, 79, 46, BLUE, "three layers add\ninterchangeable: 0.95", dy=1.5)
    arrow(59, 18, 79, 18, GREEN, "name station alone: 'seen' 0.93\nwith layer 15: 'read' 0.81", dy=1.5)
    ax.text(50, 97, "the same heads read a dog seen and a dog read, and deliver the object without its channel", ha="center", va="top", fontsize=9)
    ax.text(50, 2, "Qwen3.5-2B unless marked; the channel numbers are the 4B's (RECORD.md rows 85, 97, 99, 101 to 106)", ha="center", va="bottom", fontsize=6.5, color=GREY)
    finish(fig, out)


# ---- one-scene cards: the picture the model saw, the sentence the donor read, and the answers arm by arm


def rebuild_bank(coco_dir, crops_json):
    """the crop bank of a run from its crops.json (COCO image id and box per tile), without a model"""
    import zipfile

    from PIL import Image

    from jspace import stimuli

    C = json.load(open(crops_json))
    ann = json.load(open(Path(coco_dir) / "annotations" / "instances_val2017.json"))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    zf = zipfile.ZipFile(Path(coco_dir) / "val2017.zip")
    bank = {}
    for w, entries in C["coco"].items():
        bank[w] = []
        for iid, (x, y, bw, bh) in entries:
            im = Image.open(zf.open(f"val2017/{files[int(iid)]}")).convert("RGB")
            mx, my = bw * stimuli.MARGIN, bh * stimuli.MARGIN
            box = (max(0, int(x - mx)), max(0, int(y - my)), min(im.width, int(x + bw + mx)), min(im.height, int(y + bh + my)))
            c = im.crop(box)
            c.thumbnail((stimuli.CROP, stimuli.CROP))
            tile = Image.new("RGB", (stimuli.CROP, stimuli.CROP), stimuli.BG)
            tile.paste(c, ((stimuli.CROP - c.width) // 2, (stimuli.CROP - c.height) // 2))
            bank[w].append(tile)
    return C["vocab"], bank


def article(w):
    return "an" if w[0] in "aeiou" else "a"


def _chip(ax, x, y, text, color, w=0.16, h=0.6):
    ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, facecolor=color, alpha=0.14, lw=0))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=8, color=color)


def fig_card_cross(runs, model, out, scene=0, coco_dir=Path("data/coco2017"), run="run13"):
    """one scene of the interchange: the canvas the receiver saw, the sentence the donor read, the two answers under each arm"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from cross_modal_patch import scenes

    from jspace import stimuli

    words, bank = rebuild_bank(coco_dir, runs / run / model / "crops.json")
    S = scenes(words, bank, 64)
    r = S.iloc[scene]
    canvas = stimuli.scene(int(r.q), r.tz)
    R = read_csv(runs / run / model / "cross_modal_patch.csv")
    d = R[(R.scene == scene) & (R.direction == "typed>picture")]
    arms = ["clean", "head 5 of 19", "all heads of 19", "all heads of 7+11", "all heads of 11+15+19"]
    arms = [a for a in arms if a in set(d.arm)]
    labels = {
        "clean": "nothing written in",
        "head 5 of 19": "head 5 of layer 19",
        "all heads of 19": "all heads of layer 19",
        "all heads of 7+11": "all heads of layers 7 and 11",
        "all heads of 11+15+19": "all heads of layers 11, 15 and 19",
    }
    fig = plt.figure(figsize=(8.6, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(canvas)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s_ in ax0.spines.values():
        s_.set_visible(False)
    ax0.set_title(f"what the model sees: {article(r.Z)} {r.Z} in slot {int(r.q)}", fontsize=8.5, loc="left")
    ax0.text(0, 1.0, "", transform=ax0.transAxes)
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(arms) - 0.6, 1.6)
    ax.axis("off")
    ax.text(0.0, 1.25, f"the donor read: 'Image slot {int(r.q)} contains a {r.X}.'  (no picture)", fontsize=8.5, color=BLACK, va="center")
    ax.text(0.0, 0.5, "written into the receiver", fontsize=7.5, color=GREY, va="center")
    ax.text(0.55, 0.5, "say the name", fontsize=7.5, color=GREY, va="center", ha="center")
    ax.text(0.8, 0.5, "animal? (7/3)", fontsize=7.5, color=GREY, va="center", ha="center")
    for i, a in enumerate(arms):
        y = -i
        ax.text(0.0, y, labels.get(a, a), fontsize=8, va="center")
        for t, x in (("say", 0.47), ("animal", 0.72)):
            row = d[(d.task == t) & (d.arm == a)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            col = BLUE if row.follows_X else ORANGE if row.keeps_Z else GREY
            _chip(ax, x, y, str(row.generation)[:9], col)
    ax.text(0.0, -len(arms) - 0.25, f"blue: the answer follows the word the donor read ({r.X}); orange: it follows the picture ({r.Z})", fontsize=6.5, color=GREY, va="center")
    fig.suptitle(f"{model}, scene {scene}: the heads that read '{r.X}' are written into the model that sees {article(r.Z)} {r.Z}", x=0.01, y=1.0, ha="left", fontsize=9.5)
    finish(fig, out)


def fig_card_source(runs, model, out, scene=0, coco_dir=Path("data/coco2017"), runs_used=("run15", "run15b")):
    """one scene of the source report: the canvas, the note, the question, and 'name, seen or read' under each arm"""
    from jspace import stimuli

    words, bank = rebuild_bank(coco_dir, runs / runs_used[0] / model / "crops.json")
    frames = [read_csv(runs / r / model / "source_report.csv") for r in runs_used if (runs / r / model / "source_report.csv").exists()]
    R = pd.concat(frames)
    d = R[(R.scene == scene) & (R.target == "q")].drop_duplicates("arm")
    r = d.iloc[0]
    tz = bank[r.Z][(scene + 1) % len(bank[r.Z])]
    canvas = stimuli.scene(int(r.q), tz)
    order = ["clean", "donor"] + sorted([a for a in d.arm if a.startswith("all heads")], key=lambda a: (len(a.split("+")), a))
    labels = {"clean": "nothing written in", "donor": "the donor itself (no picture)"}
    fig = plt.figure(figsize=(8.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(canvas)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s_ in ax0.spines.values():
        s_.set_visible(False)
    ax0.set_title(f"what the model sees: {article(r.Z)} {r.Z} in slot {int(r.q)}", fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(order) - 0.6, 2.4)
    ax.axis("off")
    ax.text(0.0, 2.05, f"it also reads: 'A note says: slot {int(r.p)} contains {article(r.W)} {r.W}.'", fontsize=8.5, va="center")
    ax.text(0.0, 1.4, f"the donor read notes about slot {int(r.q)} ({article(r.X)} {r.X}) and slot {int(r.p)}, and saw no picture", fontsize=8.5, va="center")
    ax.text(0.0, 0.7, f"'what is in slot {int(r.q)}, and did you see it or read it?'", fontsize=8.5, color=BLACK, va="center", style="italic")
    for i, a in enumerate(order):
        y = -i
        ax.text(0.0, y, labels.get(a, a.replace("all heads of ", "all heads of layers " if "+" in a else "all heads of layer ").replace("+", ", ")), fontsize=8, va="center")
        row = d[d.arm == a].iloc[0]
        name = str(row.name) if isinstance(row.name, str) else str(row.generation).split(",")[0]
        ncol = BLUE if row.follows_X else ORANGE if row.keeps_Z else GREY
        _chip(ax, 0.62, y, name, ncol, w=0.17)
        src = row.source if isinstance(row.source, str) else "?"
        _chip(ax, 0.82, y, src, ORANGE if src == "seen" else BLUE if src == "read" else GREY, w=0.14)
    ax.text(0.0, -len(order) - 0.25, "name: blue follows the read word, orange the picture; channel: orange 'seen' (the picture), blue 'read' (the note)", fontsize=6.5, color=GREY, va="center")
    fig.suptitle(f"{model}, scene {scene}: does the model say it saw the {r.X} it in fact read?", x=0.01, y=1.0, ha="left", fontsize=9.5)
    finish(fig, out)


# ---- where in the question each head deposits the object (the position-resolved alignment), and the stations across models


def fig_cross_positions(runs, model, out, layers=(7, 19), question="animal", run="run13a", n_show=40):
    """per head (rows) and question token counted back from the answer (columns), how much closer the head's output is to the same
    object read than to the other object read; one panel per station layer. One question: where does each head put the object?"""
    P = read_csv(runs / run / model / "alignment_positions.csv")
    P = P[P.task == question].groupby(["layer", "head", "pos"]).agg(w=("within", "mean"), a=("across_from_picture", "mean")).reset_index()
    P["d"] = P.w - P.a
    H = int(P["head"].max()) + 1
    fig, axes = plt.subplots(len(layers), 1, figsize=(0.22 * n_show + 1.8, 0.28 * H * len(layers) + 1.4), sharex=True)
    axes = np.atleast_1d(axes)
    vmax = max(0.05, float(P.d.max()))
    for ax, L in zip(axes, layers):
        M = P[P.layer == L].pivot_table(index="head", columns="pos", values="d").reindex(columns=range(n_show)).values
        for h in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[h, j]
                if np.isnan(v):
                    continue
                ax.add_patch(plt.Rectangle((j, -h), 1, 1, facecolor=BLACK, alpha=min(0.9, max(0.0, v / vmax)), lw=0))
        ax.set_xlim(0, n_show)
        ax.set_ylim(-M.shape[0] + 1, 1)
        ax.set_yticks([-h + 0.5 for h in range(M.shape[0])])
        ax.set_yticklabels([f"head {h}" for h in range(M.shape[0])], fontsize=7)
        ax.set_title(f"layer {L}", fontsize=8.5)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xticks([j + 0.5 for j in range(0, n_show, 5)])
    axes[-1].set_xticklabels([f"-{j}" for j in range(0, n_show, 5)], fontsize=7)
    axes[-1].set_xlabel("question token, counted back from the answer position (0)")
    fig.suptitle(f"{model}: where each head puts the object it read, by question token (darker: closer to the same object read than to the other)", x=0.01, y=1.0, ha="left", fontsize=9.5)
    foot = f"the {LABEL[question]} question, 64 scenes; cosine(X seen, X read) minus cosine(X seen, Z read), mean over scenes, at each shared token; scale 0 to {vmax:.2f}"
    fig.text(0.01, -0.04, foot, fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


def fig_stations(runs, out, models=(("Qwen3.5-2B", 24), ("Qwen3.5-4B", 32), ("gemma-4-E4B-it", 42), ("Qwen3.5-27B", 64)), run="run7k"):
    """the two stations (last single reader for the decisions and for the name, by knockout) against depth, per model: do they sit at
    a share of depth or at fixed layers?"""
    rows = []
    for model, n_layers in models:
        f = runs / ("run7ka" if model == "gemma-4-31B-it" else run) / model / "summary.json"  # the 31B's every-layer run (RECORD row 128)
        if not f.exists():
            continue
        Q = json.load(open(f))["questions"]
        dec = [Q[t]["last_reader"] for t in ("animal", "category", "set", "mc") if t in Q and Q[t].get("last_reader") is not None and Q[t]["clean_correct"] >= 0.85]
        cop = [Q[t]["last_reader"] for t in ("say", "letter") if t in Q and Q[t].get("last_reader") is not None and Q[t]["clean_correct"] >= 0.85]
        rows.append((model, n_layers, dec, cop))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, share in zip(axes, (False, True)):
        for i, (_model, n, dec, cop) in enumerate(rows):
            for vals, c, lab in ((dec, ORANGE, "decisions"), (cop, BLUE, "copies")):
                y = [v / n if share else v for v in vals]
                ax.scatter([i] * len(y), y, color=c, s=28, zorder=3, label=lab if i == 0 else None)
            if not share:
                ax.plot([i - 0.3, i + 0.3], [n, n], color=GREY, lw=0.8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{m}\n({n} layers)" for m, n, _, _ in rows], fontsize=7)
        ax.set_ylabel("share of depth" if share else "layer (grey: the last layer)")
        ax.set_title("as a share of depth" if share else "as a layer", fontsize=8.5)
        if share:
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("the last layer that reads the picture for a decision (orange) and for the name (blue), per model", x=0.01, y=1.02, ha="left", fontsize=9.5)
    fig.text(0.01, -0.06, "last reader: the largest layer whose knockout from that layer on costs >= 1 nat; one point per gate-passing question; Qwen: softmax layers only", fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


# ---- which question token reads the picture (R15)


def fig_query(runs, model, out, run="run18", questions=("animal", "category", "say")):
    """per question and station, the cost in nats of blocking the picture reads of every text token, of the predicate token only,
    of the answer position only, and of every token but the predicate. One question: which token does the reading?"""
    S = json.load(open(runs / run / model / "summary.json"))
    Q = S["questions"]
    arms = ("all text tokens", "predicate token only", "answer position only", "all but the predicate")
    short = ("every text token", "the predicate only", "the answer position only", "all but the predicate")
    fig, axes = plt.subplots(1, len(questions), figsize=(2.6 * len(questions) + 1, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, questions):
        if t not in Q:
            continue
        c = BLUE if t in COPIES else ORANGE
        for k, (sname, hatch) in enumerate((("decision station", None), ("name station", "///"))):
            for j, a in enumerate(arms):
                e = Q[t]["arms"][f"{a}, {sname}"]["cost"]
                x = j + (k - 0.5) * 0.38
                ax.bar(x, e[0], width=0.36, color=c, alpha=0.9 if k == 0 else 0.45, hatch=hatch, lw=0, label=sname if j == 0 else None)
                ax.plot([x, x], [e[1], e[2]], color=BLACK, lw=0.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=7)
        ax.set_title(LABEL[t], color=c)
        ax.axhline(0, color=GREY, lw=0.5)
    axes[0].set_ylabel("cost in nats of blocking those reads")
    axes[0].legend(fontsize=6.5, loc="upper right")
    st = S["decision_station"], S["name_station"]
    fig.suptitle(f"{model}: which question token reads the picture (decision station layers {st[0]}, name station {st[1]})", x=0.01, y=1.02, ha="left", fontsize=9.5)
    fig.text(0.01, -0.08, f"n = {S['n']} scenes; the predicate is the word '{S['predicate']['animal'].strip()}' in the rule (or 'object' for the name); 95% scene bootstraps", fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


# ---- the stations across every model that has a knockout, as layer bars (the 14 October panel A, redrawn from the knockouts)


def fig_station_bars(runs, out, run="run7k"):
    """one row per model: its layers as cells (dark: softmax attention in Qwen, every layer in Gemma), with the last reader for the
    decisions (orange) and for the copies (blue) marked; x is layer / total layers. One question: where do the two stations sit?"""
    order = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "gemma-4-E4B-it", "gemma-4-31B-it"]
    rows = []
    for model in order:
        f = runs / run / model / "summary.json"
        if model == "gemma-4-31B-it":  # the 31B's first two grids missed its readers (RECORD rows 123, 124); only the every-layer run counts
            f = runs / "run7ka" / model / "summary.json"
        if not f.exists():
            continue
        S = json.load(open(f))
        Q, AL, n = S["questions"], S["attention_layers"], S["n_layers"] or (max(S["attention_layers"]) + 1)
        dec = sorted({Q[t]["last_reader"] for t in ("animal", "category", "set", "mc", "yesno") if t in Q and Q[t].get("last_reader") is not None and Q[t]["clean_correct"] >= 0.85})
        cop = sorted({Q[t]["last_reader"] for t in ("say", "letter") if t in Q and Q[t].get("last_reader") is not None and Q[t]["clean_correct"] >= 0.85})
        rows.append((model, n, AL, dec, cop))
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 1.6))
    for i, (model, n, AL, dec, cop) in enumerate(rows):
        y = -i
        blocked = set(AL)
        for L in range(n):
            ax.add_patch(plt.Rectangle((L / n, y - 0.3), 1 / n, 0.6, facecolor=GREY if L in blocked else "#e8e8e8", alpha=0.75 if L in blocked else 1.0, lw=0.3, edgecolor="white"))
        for vals, c in ((dec, ORANGE), (cop, BLUE)):
            for v in vals:
                ax.plot([(v + 1) / n, (v + 1) / n], [y - 0.42, y + 0.42], color=c, lw=2.2, solid_capstyle="butt")
        ax.text(-0.012, y, model.replace("gemma-4-", "Gemma ").replace("-it", "").replace("Qwen3.5-", "Qwen "), ha="right", va="center", fontsize=8)
        ax.text(1.012, y, f"{n} layers", ha="left", va="center", fontsize=7, color=GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(rows) + 0.5, 0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("layer / total layers")
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([], [], color=ORANGE, lw=2.2, label="last layer that reads the picture for a decision")
    ax.plot([], [], color=BLUE, lw=2.2, label="last layer that reads it for the name or its letter")
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    foot = (
        "by attention knockout (the largest layer whose block from that layer on costs >= 1 nat), one mark per gate-passing question;\n"
        "dark cells: layers whose reads can be blocked (softmax attention); 64 scenes per model"
    )
    fig.text(0.01, -0.14, foot, fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


# ---- how the picture's read is distributed over depth (post hoc, 21 Oct): the from-L-on cost as a share of the all-layers cost


def access_profiles(runs, run="run7k", decisions=("animal", "category", "set", "mc", "yesno"), copies=("say", "letter")):
    """per model and question class, the remaining share of the read when every blockable layer from L on is blocked, against L / layers;
    with the depth at which half the read is done and the width between 0.9 and 0.1 remaining"""
    order = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "gemma-4-E4B-it", "gemma-4-31B-it"]
    out = []
    for model in order:
        f = runs / ("run7ka" if model == "gemma-4-31B-it" else run) / model / "summary.json"  # the 31B's every-layer run (RECORD row 128)
        if not f.exists():
            continue
        S = json.load(open(f))
        Q, n = S["questions"], S["n_layers"] or (max(S["attention_layers"]) + 1)
        for cls, names in (("decisions", decisions), ("copies", copies)):
            for t in names:
                q = Q.get(t)
                if not q or q["clean_correct"] < 0.85 or not q.get("from"):
                    continue
                Ls = sorted(int(k) for k in q["from"])
                total = q["from"][str(Ls[0])]["cost"][0]
                if total < 1:  # a read the knockout cannot measure (RECORD rows 123, 124)
                    continue
                x = np.array([L / n for L in Ls])
                y = np.clip(np.array([q["from"][str(L)]["cost"][0] / total for L in Ls]), 0, 1)
                peak = max(v["cost"][0] for v in q["single"].values()) / total  # the largest single layer's share of the whole read
                out.append(dict(model=model, n_layers=n, cls=cls, question=t, x=x, y=y, peak=peak))
    return out


def _crossing(x, y, level):
    """the first x at which the (decreasing) profile y falls to `level`, by linear interpolation"""
    for i in range(1, len(x)):
        if y[i] <= level:
            if y[i - 1] == y[i]:
                return float(x[i])
            return float(x[i - 1] + (x[i] - x[i - 1]) * (y[i - 1] - level) / (y[i - 1] - y[i]))
    return float(x[-1])


def fig_access(runs, out, run="run7k"):
    """one panel per model: the share of the picture's read still to come when every blockable layer from L on is blocked, for the
    decisions (orange) and the copies (blue). One question: is the read a step at a station or a slope over many layers, per model?"""
    P = access_profiles(runs, run)
    models = list(dict.fromkeys(p["model"] for p in P))
    fig, axes = plt.subplots(1, len(models), figsize=(1.55 * len(models) + 0.8, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, model in zip(axes, models):
        n = next(p["n_layers"] for p in P if p["model"] == model)
        for cls, c in (("decisions", ORANGE), ("copies", BLUE)):
            ps = [p for p in P if p["model"] == model and p["cls"] == cls]
            if not ps:
                continue
            grid = np.linspace(0, 1, 201)
            Y = np.array([np.interp(grid, p["x"], p["y"]) for p in ps])
            m = Y.mean(0)
            ax.fill_between(grid, Y.min(0), Y.max(0), color=c, alpha=0.18, lw=0)
            ax.plot(grid, m, color=c, lw=1.6)
            half, w = _crossing(grid, m, 0.5), _crossing(grid, m, 0.1) - _crossing(grid, m, 0.9)
            peak = float(np.mean([p["peak"] for p in ps]))
            rows.append(dict(model=model, n_layers=n, cls=cls, questions=len(ps), half_depth=round(half, 3), width=round(w, 3), width_layers=round(w * n, 1), peak_layer_share=round(peak, 2)))
            ax.plot([half, half], [0, 0.5], color=c, lw=0.7, ls=":")
        ax.set_title(f"{model.replace('gemma-4-', 'Gemma ').replace('-it', '').replace('Qwen3.5-', 'Qwen ')}\n({n} layers)", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("L / layers", fontsize=8)
    axes[0].set_ylabel("share of the read after L")
    axes[0].plot([], [], color=ORANGE, lw=1.6, label="decisions")
    axes[0].plot([], [], color=BLUE, lw=1.6, label="the name, its letter")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("the share of the picture's read still to come after layer L: the decisions' read ends before the name's in every model", x=0.01, y=1.04, ha="left", fontsize=9.5)
    foot = (
        "y: the knockout cost of blocking every blockable layer from L on, as a share of the cost of blocking them all; mean over gate-passing questions, band min to max;\n"
        "dotted: the depth by which half the read is done; 64 scenes per model; Qwen blocks its softmax layers, Gemma every layer"
    )
    fig.text(0.01, -0.16, foot, fontsize=6.5, color=GREY)
    fig.tight_layout()
    T = pd.DataFrame(rows)
    T.to_csv(str(out).replace(".png", ".csv"), index=False)
    print(T.to_string(index=False))
    finish(fig, out)


# ---- the matched-disruption source test (R23): does the report follow the pathway written in, or the size of the disruption?


def fig_source_cause(runs, model, out, run="run20"):
    """left: per station group, the share of scenes on which the report says "read" after a donor's heads are written in, the
    picture present and unblocked: the note-cause donor (blue), another picture of the same object (orange), noise of the
    note-cause donor's size (grey), the receiver itself (light grey). Right: the size of each donor's displacement per layer."""
    S = json.load(open(runs / run / model / "summary.json"))
    groups = ["+".join(map(str, g)) for g in S["groups"]]
    donors = (
        ("A", BLUE, "the note's heads (empty canvas, same note)"),
        ("B", ORANGE, "another picture of the same object"),
        ("N", GREY, "noise of the note donor's size"),
        ("C", "#cfcfcf", "the receiver's own heads (null patch)"),
    )
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for j, g in enumerate(groups):
        for k, (d, c, lab) in enumerate(donors):
            key = f"{d}: all heads of {g}"
            if key not in S["arms"]:
                continue
            e = S["arms"][key]["read"]
            x = j + (k - 1.5) * 0.2
            ax.bar(x, e[0], width=0.19, color=c, lw=0, label=lab if j == 0 else None)
            ax.plot([x, x], [e[1], e[2]], color=BLACK, lw=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"layers {g}" for g in groups], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('share of scenes reporting "read"')
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("the picture present and unblocked; the answer the same on every scene", fontsize=8)
    for d, c, _lab in donors[:3]:
        Ls = sorted(int(k) for k in S["displacement"][d])
        ax2.plot(Ls, [S["displacement"][d][str(L)] for L in Ls], color=c, marker="o", ms=3.5, lw=1.2)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("relative displacement of the heads' outputs", fontsize=7.5)
    ax2.set_ylim(0)
    ax2.set_title("how far each donor moves the outputs", fontsize=8)
    fig.suptitle(f"{model}: the report follows which pathway's heads were written in, with the picture present and unblocked", x=0.01, y=1.03, ha="left", fontsize=9.5)
    gates = ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in S["gates"].items())
    fig.text(0.01, -0.1, f"n = {S['n']} scenes; 95% scene bootstraps; gates: {gates}", fontsize=6.5, color=GREY)
    fig.tight_layout()
    finish(fig, out)


# ---- two real sources in conflict (R14): does the report follow the cause?


def fig_conflict(runs, models, out, run="run17", congruent=False):
    """per arm, the share of scenes naming the note's object (blue) or the picture's (orange), split by the reported channel (filled:
    the report matches the named object's channel; hatched: it does not). One question: when one channel is blocked, does the report
    move with the answer?"""
    short = {
        "clean": "nothing blocked",
        "picture blocked": "picture blocked",
        "note blocked": "note blocked",
        "picture blocked at the decision station": "picture, decision station",
        "picture blocked at the name station": "picture, name station",
        "note blocked at the decision station": "note, decision station",
        "note blocked at the name station": "note, name station",
        "both blocked": "both blocked",
    }
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        f = runs / run / model / "source_conflict.csv"
        if not f.exists():
            continue
        R = read_csv(f)
        order = [a for a in short if a in set(R.arm)]
        for i, a in enumerate(order):
            g = R[R.arm == a]
            n = len(g)
            x_read = ((g.names_X) & (g.source == "read")).sum() / n
            x_seen = ((g.names_X) & (g.source == "seen")).sum() / n
            z_seen = ((g.names_Z) & (g.source == "seen")).sum() / n
            z_read = ((g.names_Z) & (g.source == "read")).sum() / n
            b = 0
            for v, c, h in ((x_read, BLUE, None), (x_seen, BLUE, "///"), (z_seen, ORANGE, None), (z_read, ORANGE, "///")):
                ax.bar(i, v, bottom=b, width=0.64, color=c, alpha=0.9 if h is None else 0.45, hatch=h, lw=0)
                b += v
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[a] for a in order], rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{model} (n = {R.scene.nunique()})", pad=6)
    axes[0].set_ylabel("share of scenes")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.45, hatch="///"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.45, hatch="///"),
    ]
    names = ["names the note's object, says 'read'", "names the note's object, says 'seen'", "names the picture's object, says 'seen'", "names the picture's object, says 'read'"]
    fig.legend(handles, names, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=4, frameon=False, handlelength=1.4, columnspacing=1.2)
    title = (
        "a picture and a note naming the same object in the same slot: the answer is the same from either, which channel is reported"
        if congruent
        else "a picture of one object and a note naming another in the same slot: what is named, and which channel is reported"
    )
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=9.5)
    foot = (
        "the picture's reads: attention from the question to the image tokens; the note's: to the note word's tokens;\n"
        "blocked at every blockable layer (every softmax layer in Qwen, every layer in Gemma), at the decision station, or at the name station"
    )
    fig.text(0.01, 0.01, foot, fontsize=6.5, color=GREY)
    fig.subplots_adjust(top=0.78, bottom=0.4, wspace=0.12, left=0.07, right=0.99)
    finish(fig, out)


def fig_card_conflict(runs, model, out, scene=0, coco_dir=Path("data/coco2017"), run="run17"):
    """one scene of the conflict: the canvas, the note, the question, and 'name, channel' under each arm"""
    from jspace import stimuli

    words, bank = rebuild_bank(coco_dir, runs / run / model / "crops.json")
    R = read_csv(runs / run / model / "source_conflict.csv")
    d = R[R.scene == scene]
    r = d.iloc[0]
    tz = bank[r.Z][(scene + 1) % len(bank[r.Z])]
    canvas = stimuli.scene(int(r.q), tz)
    order = ["clean", "picture blocked", "picture blocked at the decision station", "picture blocked at the name station", "note blocked", "both blocked"]
    order = [a for a in order if a in set(d.arm)]
    labels = {
        "clean": "nothing blocked",
        "picture blocked": "picture blocked",
        "picture blocked at the decision station": "picture blocked, decision station",
        "picture blocked at the name station": "picture blocked, name station",
        "note blocked": "note blocked",
        "both blocked": "both blocked",
    }
    fig = plt.figure(figsize=(8.6, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.08)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(canvas)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for s_ in ax0.spines.values():
        s_.set_visible(False)
    ax0.set_title(f"what the model sees: {article(r.Z)} {r.Z} in slot {int(r.q)}", fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(order) - 0.6, 2.2)
    ax.axis("off")
    congruent = bool(d.names_Z.any()) and not bool(d.names_X.any())
    noted = r.Z if congruent else r.X
    ax.text(0.0, 1.85, f"it also reads: 'A note says: slot {int(r.q)} contains {article(noted)} {noted}.'", fontsize=8.5, va="center")
    ax.text(0.0, 1.05, f"'what is the object in slot {int(r.q)}, and did you see it or read it?'", fontsize=8.5, va="center", style="italic")
    for i, a in enumerate(order):
        y = -i
        ax.text(0.0, y, labels[a], fontsize=8, va="center")
        row = d[d.arm == a].iloc[0]
        name = str(row["name"]) if isinstance(row["name"], str) else str(row.generation).split(",")[0]
        ncol = BLUE if row.names_X else ORANGE if row.names_Z else GREY
        _chip(ax, 0.64, y, name, ncol, w=0.16)
        src = row.source if isinstance(row.source, str) else "?"
        _chip(ax, 0.83, y, src, ORANGE if src == "seen" else BLUE if src == "read" else GREY, w=0.13)
    ax.text(0.0, -len(order) - 0.25, "name: blue the note's object, orange the picture's; channel: orange 'seen', blue 'read'", fontsize=6.5, color=GREY, va="center")
    what = f"{article(r.Z)} {r.Z} in the picture and the same {r.Z} in the note" if congruent else f"{article(r.Z)} {r.Z} in the picture, {article(r.X)} {r.X} in the note"
    fig.suptitle(f"{model}, scene {scene}: {what}; what it names and which channel it reports", x=0.01, y=1.0, ha="left", fontsize=9.5)
    finish(fig, out)
