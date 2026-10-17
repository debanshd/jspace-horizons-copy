#!/usr/bin/env python
"""The eight worked examples: the inputs a model received and the answers it wrote, one figure each.

Every input is cut from the stimulus sheet a run saved (stimuli.png pastes each canvas at full size, so the crop is the canvas pixel
for pixel) or drawn with the package's own stimulus code (the empty and printed-word canvases, the SCAM and GQA photos), and the
script checks the drawing against the saved sheets. Every answer is read from the runs' CSVs; nothing is re-run. The rule
questions asked for 7 (yes) or 3 (no); the figures print yes and no, and outputs.csv keeps every answer verbatim beside it.

    uv run scripts/make_figures.py                       # from your runs: runs/ -> runs/examples/
    uv run scripts/make_figures.py --runs results        # from the original runs: results/ -> results/examples/

Needs data/scam and data/gqa (scripts/download.py --data-only). Figures use TeX Gyre Heros when it is installed (apt package fonts-texgyre).
"""

import argparse
import glob
import logging
import os
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

from jspace import stimuli
from jspace.config import DATA_DIR, RUNS_DIR

RUNS, DATA, OUT = str(RUNS_DIR), str(DATA_DIR), str(RUNS_DIR / "examples")  # set from the command line in main()
FALLBACK = None  # --fallback: where to read a run that is missing from RUNS (e.g. results/ for the Gemma runs)
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def tint(c, a):
    c = c.lstrip("#")
    return "#" + "".join(f"{int(255 - a * (255 - int(c[i : i + 2], 16))):02X}" for i in (0, 2, 4))


ROWS = []  # every answer shown, for outputs.csv


def run_file(rel):
    """a file of a run: from RUNS, or from FALLBACK when that run has not been done there (said so on the console)"""
    if not os.path.exists(f"{RUNS}/{rel}") and FALLBACK and os.path.exists(f"{FALLBACK}/{rel}"):
        print(f"{rel}: not in {RUNS}, read from {FALLBACK}")
        return f"{FALLBACK}/{rel}"
    return f"{RUNS}/{rel}"


def scam_photo(fn):
    """a SCAM photo as the model received it"""
    return stimuli.fit_photo(f"{DATA}/scam/images/train/{fn}")


def slot(canvas, s, n_cols=3):
    r, c = divmod(s - 1, n_cols)
    return canvas.crop((c * stimuli.SLOT, r * stimuli.SLOT, (c + 1) * stimuli.SLOT, (r + 1) * stimuli.SLOT))


# --- reading the logs
def truthy(v):
    return v is True or str(v) == "True"


def one(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    assert m.sum() == 1, (kw, int(m.sum()))
    return df[m].iloc[0]


def shown(task, g):
    """what a figure prints: the rule answers as yes / no (the prompt asked for 7 / 3), everything else as generated, trimmed"""
    g = str(g).strip()
    if task == "animal then name":  # "7. Train" is printed as "yes, train"
        return {"7": "yes", "3": "no"}.get(g[:1], g[:1]) + ", " + g[1:].strip(" .,").lower()
    if task in ("animal", "ride", "set"):
        return {"7": "yes", "3": "no"}.get(g.strip("*.")[:1], g)
    return g


def whose(g, x_truth, z_truth):
    g = str(g).strip().strip("*").lower()
    gx, gz = g.startswith(str(x_truth).lower()), g.startswith(str(z_truth).lower())
    return "X" if gx and not gz else "Z" if gz and not gx else "N"


def kind_of(row):
    return "X" if truthy(row.follows_X) else "Z" if truthy(row.keeps_Z) else "N"


def log(example, model, data, inputs, prompt, intervention, source, filt, task, verbatim, kind):
    ROWS.append(
        dict(
            example=example,
            model=model,
            data=data,
            input_images=inputs,
            prompt=prompt,
            intervention=intervention,
            task=task,
            output_verbatim=str(verbatim).strip(),
            output_shown=shown(task, verbatim),
            whose_answer=kind,
            source_csv=source,
            source_row=filt,
        )
    )


# --- drawing, in the project's figure style: TeX Gyre Heros, white ground, grey text, blue = naming, orange = a rule or the fact, grey = the rest
BLUE, ORANGE = "#0072B2", "#D55E00"
W_IN = 13.33  # every figure is one slide wide


def G(v):
    return "#%02X%02X%02X" % ((round(255 * v),) * 3)


T_TITLE, T_LABEL, T_SEC = G(0.1), G(0.2), G(0.4)
F_BLUE, F_ORANGE, F_OWN = tint(BLUE, 0.30), tint(ORANGE, 0.30), G(0.945)
FS = dict(title=17, sub=11.5, label=12.5, cell=13, col=11, key=12, panel=13, tile=12, arrow=11)
IMG = {}  # name -> PIL image at full size, saved to inputs/


def style():
    for f in glob.glob("/usr/share/**/*[Hh]eros*.[ot]tf", recursive=True):
        fm.fontManager.addfont(f)
    font = next((f for f in ["TeX Gyre Heros", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"] if f in {x.name for x in fm.fontManager.ttflist}), "DejaVu Sans")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    sns.set_theme(
        context="paper",
        style="ticks",
        palette=OKABE,
        rc={
            "font.family": [font],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "normal",
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
    )
    plt.rcParams.update({"font.family": "TeX Gyre Heros", "font.size": 12, "svg.fonttype": "none"})


class Page:
    """positions in inches from the top of a figure W_IN wide and h tall, turned into figure fractions"""

    def __init__(self, h):
        self.h = h
        self.fig = plt.figure(figsize=(W_IN, h))

    def x(self, inches):
        return inches / W_IN

    def y(self, top):
        return 1 - top / self.h

    def axes(self, x, top, w, h):
        return self.fig.add_axes([x / W_IN, 1 - (top + h) / self.h, w / W_IN, h / self.h])


def fill(kind, fam):
    """a box is coloured only when the answer follows the moved or added object: blue for naming, orange for a rule"""
    return (F_BLUE if fam == "name" else F_ORANGE) if kind == "X" else F_OWN


def ink(kind):
    return T_TITLE if kind == "X" else G(0.35)


def header(p, title, sub):
    p.fig.text(0.0, p.y(0.0), title, fontsize=FS["title"], color=T_TITLE, ha="left", va="bottom")
    p.fig.text(0.0, p.y(0.1), sub, fontsize=FS["sub"], color=T_SEC, ha="left", va="top")


def show(ax, im, label=None, anchor="C", label_loc="center"):
    ax.imshow(np.asarray(im), interpolation="lanczos")
    ax.set_anchor(anchor)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(G(0.8))
        s.set_linewidth(0.7)
    if label:
        ax.set_title(label, fontsize=FS["tile"], color=T_LABEL, pad=6, loc=label_loc)


def text_tile(ax, text, label=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0,rounding_size=0.05", facecolor=G(0.965), edgecolor=G(0.8), lw=0.7))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=FS["tile"], color=T_TITLE, linespacing=1.4)
    if label:
        ax.set_title(label, fontsize=FS["tile"], color=T_LABEL, pad=6, loc="center")


def arrow(p, x0, x1, top, label=None):
    y = p.y(top)
    p.fig.add_artist(matplotlib.patches.FancyArrowPatch((p.x(x0), y), (p.x(x1), y), transform=p.fig.transFigure, arrowstyle="-|>", mutation_scale=14, color=G(0.55), lw=1.2))
    if label:
        p.fig.text(p.x((x0 + x1) / 2), y + 0.09 / p.h * 2, label, ha="center", va="bottom", fontsize=FS["arrow"], color=T_SEC, linespacing=1.3)


def labels(p, x, top, items, across=None, step=0.3):
    """direct colour labels in plain words: one per line, or along one line with `across` inches between starts"""
    for i, (c, txt) in enumerate(items):
        p.fig.text(p.x(x + (i * across if across else 0)), p.y(top + (0 if across else i * step)), txt, color=c, fontsize=FS["key"], ha="left", va="top")


def panel(p, top, letter, txt):
    p.fig.text(0.0, p.y(top), letter, fontsize=FS["panel"], color=T_TITLE, ha="left", va="top", weight="bold")
    p.fig.text(p.x(0.24), p.y(top), txt, fontsize=FS["panel"], color=T_TITLE, ha="left", va="top")


def grid(ax, text, kinds, fams, row_labels, col_labels, fs=None, col_note=None, two=None, brackets=(), seps=()):
    """answers in boxes: row i, column j holds text[i][j]; two[(i, j)] = ((text, kind, fam), (text, kind, fam)) splits a box in two"""
    fs = fs or FS["cell"]
    nr, nc = len(text), len(text[0])
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.axis("off")
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i in range(nr):
        for j in range(nc):
            if two and (i, j) in two:
                (lt, lk, lf), (rt, rk, rf) = two[(i, j)]
                ax.add_patch(Rectangle((j + 0.03, i + 0.09), 0.47, 0.82, facecolor=fill(lk, lf), edgecolor="none"))
                ax.add_patch(Rectangle((j + 0.50, i + 0.09), 0.47, 0.82, facecolor=fill(rk, rf), edgecolor="none"))
                ax.text(j + 0.485, i + 0.5, lt, ha="right", va="center", fontsize=fs, color=ink(lk))
                ax.text(j + 0.515, i + 0.5, rt, ha="left", va="center", fontsize=fs, color=ink(rk))
                continue
            k = kinds[i][j]
            ax.add_patch(Rectangle((j + 0.03, i + 0.09), 0.94, 0.82, facecolor=fill(k, fams[i][j]), edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, text[i][j], ha="center", va="center", fontsize=fs, color=ink(k))
        ax.text(-0.01, i + 0.5, row_labels[i], ha="right", va="center", fontsize=FS["label"], color=T_LABEL, transform=tr, clip_on=False)
    for j, c in enumerate(col_labels):
        ax.text(j + 0.5, -0.14, c, ha="center", va="bottom", fontsize=FS["col"], color=T_SEC, clip_on=False, linespacing=1.15)
    if col_note:
        ax.text(-0.01, -0.14, col_note, ha="right", va="bottom", fontsize=FS["col"], color=T_SEC, transform=tr, clip_on=False)
    for j0, j1, lab in brackets:
        y = -1.0 if any("\n" in c for c in col_labels) else -0.7
        ax.plot([j0 + 0.05, j0 + 0.05, j1 + 0.95, j1 + 0.95], [y + 0.13, y, y, y + 0.13], color=G(0.55), lw=0.9, clip_on=False)
        ax.text((j0 + j1 + 1) / 2, y - 0.08, lab, ha="center", va="bottom", fontsize=FS["key"] - 0.5, color=T_LABEL, clip_on=False)
    for i0 in seps:
        ax.plot([0.03, nc - 0.03], [i0, i0], color=G(0.8), lw=0.8, clip_on=False)
    return tr


def regions(kinds, name_rows, rule_rows, labels, n_first=1):
    """brackets over runs of columns where every answer follows the patched object (A), where every name does and every rule
    answers for the seen object (B), and where every answer is for the seen object (C); mixed columns get no bracket"""
    tag = []
    for j in range(n_first, len(kinds[0])):
        col = [kinds[i][j] for i in range(len(kinds))]
        tag.append(
            "A" if all(k == "X" for k in col) else "C" if all(k == "Z" for k in col) else "B" if all(kinds[i][j] == "X" for i in name_rows) and all(kinds[i][j] == "Z" for i in rule_rows) else None
        )
    out, j = [], 0
    while j < len(tag):
        if tag[j] is None:
            j += 1
            continue
        k = j
        while k + 1 < len(tag) and tag[k + 1] == tag[j]:
            k += 1
        out.append((j + n_first, k + n_first, labels[tag[j]]))
        j = k + 1
    assert any(lab == labels["B"] for *_, lab in out), "no column where the name follows and the rules do not"
    return out


def save(p, name):
    os.makedirs(OUT, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.2, facecolor="white")
    p.fig.savefig(f"{OUT}/{name}.png", dpi=220, **kw)
    p.fig.savefig(f"{OUT}/{name}.pdf", **kw)
    p.fig.savefig(f"{OUT}/{name}.svg", dpi=220, **kw)
    svg = open(f"{OUT}/{name}.svg").read()  # live text: fall back to Helvetica or Arial where Heros is missing
    open(f"{OUT}/{name}.svg", "w").write(svg.replace("'TeX Gyre Heros'", "'TeX Gyre Heros', Helvetica, Arial, sans-serif"))
    plt.close(p.fig)
    print("wrote", f"{OUT}/{name}.png / .pdf / .svg")


INTRO = "Study the scene. It has an image grid with numbered slots."
Q = {
    "say": "What is the object in image slot {q}? Reply with the noun only.",
    "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
    "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
    "set": "Rule: if the object in image slot {q} is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.",
    "animal then name": "Is the object in image slot {q} an animal, and what is it? Reply with 7 if it is an animal or 3 if it is not, then the noun.",
}
LABEL = {"say": "What is it?", "letter": "First letter?", "animal": "Is it an animal?", "ride": "Is it ridden or driven?", "set": "Is it a train, bird or clock?"}
FAM = {"say": "name", "letter": "name", "animal": "rule", "ride": "rule", "set": "rule"}
CANVAS = {}


def canvases():
    F4_SHEET = run_file("subspace_patching/Qwen3.5-4B/stimuli.png")  # frames: donor scene 0, donor scene 1, receiver scene 0, receiver scene 1
    CANVAS["scene0_donor_bird"] = stimuli.sheet_frame(F4_SHEET, 0, 4, 672, 448)
    CANVAS["scene1_donor_train"] = stimuli.sheet_frame(F4_SHEET, 1, 4, 672, 448)
    CANVAS["scene0_receiver_airplane"] = stimuli.sheet_frame(F4_SHEET, 2, 4, 672, 448)
    CANVAS["scene1_receiver_cat"] = stimuli.sheet_frame(F4_SHEET, 3, 4, 672, 448)
    CANVAS["scene1_donor_printed_train"] = stimuli.canvas(3, 2, {5: stimuli.word_tile("train")})
    CANVAS["g1_scene0_dog"] = stimuli.sheet_frame(run_file("name_steering/Qwen3.5-4B/stimuli.png"), 0, 4, 672, 448)
    CANVAS["empty"] = stimuli.canvas(3, 2, {})
    gs = run_file("subspace_patching/gemma-4-31B-it/stimuli.png")
    for j, n in [(0, "scene0_donor_bird"), (2, "scene0_receiver_airplane")]:  # Gemma-4-31B received the same scene-0 canvases
        assert np.array_equal(np.asarray(CANVAS[n]), np.asarray(stimuli.sheet_frame(gs, j, 4, 672, 448))), n
    assert np.array_equal(np.asarray(CANVAS["empty"])[:, 224:], np.asarray(CANVAS["scene0_donor_bird"])[:, 224:]), "e_canvas differs from the saved canvas"
    g1 = run_file("name_steering/Qwen3.5-4B/stimuli.png")
    assert np.array_equal(np.asarray(stimuli.canvas(3, 2, {3: stimuli.word_tile("dog")})), np.asarray(stimuli.sheet_frame(g1, 2, 4, 672, 448))), "t1_word_tile differs"


def win_cols(wins):
    return ["not\npatched"] + ["input" if w == -1 else str(w) for w in wins]


def strip(p, centre, top, items, tile=1.3, gap=0.14):
    """a row of input tiles and arrows centred on `centre` inches: items = [("img", PIL, label) | ("text", str, label, width) | ("arrow", label, width)]"""
    widths = [tile if it[0] == "img" else it[3] if it[0] == "text" else it[2] for it in items]
    x = centre - (sum(widths) + gap * (len(items) - 1)) / 2
    for it, w in zip(items, widths):
        if it[0] == "img":
            show(p.axes(x, top, w, tile), it[1], it[2])
        elif it[0] == "text":
            text_tile(p.axes(x, top, w, tile), it[1], it[2])
        else:
            arrow(p, x + 0.05, x + w - 0.05, top + tile / 2, it[1])
        x += w + gap


def KEY(obj, seen):
    return [(BLUE, f"blue: names the {obj}"), (ORANGE, f"orange: treats it as the {obj}"), (G(0.45), f"grey: treats it as the {seen} it sees")]


def REG(obj, seen):
    return {"A": f"everything follows the {obj}", "B": f"names the {obj}, treats it as the {seen}", "C": "no effect"}


LAYERS_NOTE = "patched at 4 layers, starting at layer"


# ---------------------------------------------------------------- ex1: one scene, five questions, two families (D1)
def ex1():
    def table_of(path, tasks, model, last):
        d = pd.read_csv(run_file(path))
        d = d[(d.scene == 0) & d.kind.isin(["none", "transplant"])]
        assert set(d.X) == {"bird"} and set(d.Z) == {"airplane"}
        wins = sorted(d[d.kind == "transplant"].w0.unique())
        text, kinds, fams = [], [], []
        for t in tasks:
            rows = [one(d, task=t, kind="none")] + [one(d, task=t, kind="transplant", w0=w) for w in wins]
            text.append([shown(t, r.generation) for r in rows])
            kinds.append([kind_of(r) for r in rows])
            fams.append([FAM[t]] * len(rows))
            for r in rows:
                iv = "not patched" if r.kind == "none" else f"the bird run's image-token states patched at layers {r.w0} to {min(r.w0 + 3, last)}"
                log(
                    "ex1",
                    model,
                    "COCO photo in slot 1 of a six-slot canvas",
                    "inputs/scene0_receiver_airplane.png; donor inputs/scene0_donor_bird.png",
                    f"{INTRO} [image] " + Q[t].format(q=1, a="train", b="bird", c="clock"),
                    iv,
                    f"runs/{path}",
                    f"scene=0, task={t}, arm={r.arm}",
                    t,
                    r.generation,
                    kind_of(r),
                )
        return wins, text, kinds, fams

    T4 = ["say", "letter", "animal", "ride", "set"]
    w4, t4, k4, f4 = table_of("depth_sweep_spectrum/Qwen3.5-4B/depth_sweep.csv", T4, "Qwen3.5-4B", 31)
    wg, tg, kg, fg = table_of("depth_sweep/gemma-4-31B-it/depth_sweep.csv", ["say", "animal"], "Gemma-4-31B", 59)
    style()
    p = Page(8.1)
    header(
        p,
        "Patched into the middle layers, the bird is named but not treated as a bird",
        "Residual-stream patching: the bird run's image-token states replace the airplane run's at 4 consecutive layers  \u00b7  Qwen3.5-4B, Gemma-4-31B",
    )
    labels(p, 0.0, 0.85, KEY("bird", "airplane"))
    tx, tw = 3.3, W_IN - 3.3  # the tables span 3.3 in to the right edge
    strip(
        p,
        tx + tw / 2,
        0.75,
        [("img", slot(CANVAS["scene0_donor_bird"], 1), "donor sees"), ("arrow", "residual stream\npatched in", 1.4), ("img", slot(CANVAS["scene0_receiver_airplane"], 1), "receiver sees")],
    )
    panel(p, 2.55, "a", "Qwen3.5-4B, 32 layers")
    grid(p.axes(tx, 3.35, tw, 2.6), t4, k4, f4, [LABEL[t] for t in T4], win_cols(w4), col_note=LAYERS_NOTE, brackets=regions(k4, [0, 1], [2, 3, 4], REG("bird", "airplane")))
    panel(p, 6.2, "b", "Gemma-4-31B, 60 layers")
    grid(p.axes(tx, 7.0, tw, 1.04), tg, kg, fg, [LABEL[t] for t in ["say", "animal"]], win_cols(wg), fs=10, col_note=LAYERS_NOTE, brackets=regions(kg, [0], [1], REG("bird", "airplane")))
    save(p, "ex1_depth_one_scene")


# ---------------------------------------------------------------- ex2: the same depths for a photo, a printed word and a typed sentence (D1)
def ex2():
    d = pd.read_csv(run_file("depth_sweep/Qwen3.5-4B/depth_sweep.csv"))
    d = d[d.scene == 1]
    assert set(d.X) == {"train"} and set(d.Z) == {"cat"}
    wins = sorted(d[d.kind == "transplant"].w0.unique())
    text, kinds, fams, labs = [], [], [], []
    srcs = [
        ("photo", "none", "transplant", "inputs/scene1_receiver_cat.png; donor inputs/scene1_donor_train.png"),
        ("printed word", "none", "transplant printed", "inputs/scene1_receiver_cat.png; donor inputs/scene1_donor_printed_train.png"),
        ("typed sentence", "typed none", "transplant typed", "none (text only)"),
    ]
    for src, clean_kind, kind, ins in srcs:
        for t in ["say", "animal"]:
            rows = [one(d, task=t, kind=clean_kind)] + [one(d, task=t, kind=kind, w0=w) for w in wins]
            text.append([shown(t, r.generation) for r in rows])
            kinds.append([kind_of(r) for r in rows])
            fams.append([FAM[t]] * len(rows))
            labs.append(LABEL[t])
            for r in rows:
                pr = (f"{INTRO} Image slot 5 contains a cat. " + Q[t].format(q=5)) if src == "typed sentence" else f"{INTRO} [image] " + Q[t].format(q=5)
                iv = (
                    "not patched"
                    if "none" in r.kind
                    else f"the state of ' train' from '{INTRO} Image slot 5 contains a train.' patched into ' cat' at layers {r.w0} to {min(r.w0 + 3, 31)}"
                    if src == "typed sentence"
                    else f"the {src} run's image-token states patched at layers {r.w0} to {min(r.w0 + 3, 31)}"
                )
                log("ex2", "Qwen3.5-4B", src, ins, pr, iv, "runs/depth_sweep/Qwen3.5-4B/depth_sweep.csv", f"scene=1, task={t}, arm={r.arm}", t, r.generation, kind_of(r))
    style()
    p = Page(7.1)
    header(
        p,
        "The same pattern whether the train is a photo, a printed word or a typed sentence",
        "Residual-stream patching: the train run's states replace the cat run's at 4 consecutive layers  \u00b7  Qwen3.5-4B",
    )
    labels(p, 0.0, 0.85, KEY("train", "cat"))
    tx, tw = 3.3, W_IN - 3.3
    strip(
        p,
        tx + tw / 2,
        0.75,
        [
            ("img", slot(CANVAS["scene1_donor_train"], 5), "photo"),
            ("img", slot(CANVAS["scene1_donor_printed_train"], 5), "printed word"),
            ("text", "'Image slot 5\ncontains a train.'", "typed sentence", 1.75),
            ("arrow", "patched\nin", 1.0),
            ("img", slot(CANVAS["scene1_receiver_cat"], 5), "receiver sees"),
            ("text", "'Image slot 5\ncontains a cat.'", "or reads", 1.6),
        ],
        tile=1.2,
    )
    ax = p.axes(tx, 3.15, tw, 3.85)
    grid(ax, text, kinds, fams, labs, win_cols(wins), col_note=LAYERS_NOTE, seps=(2, 4), brackets=regions(kinds, [0, 2, 4], [1, 3, 5], REG("train", "cat")))
    trf = blended_transform_factory(p.fig.transFigure, ax.transData)
    for i0, lab in [(0, "photo"), (2, "printed word"), (4, "typed sentence")]:
        ax.text(0.0, i0 + 1.0, lab, ha="left", va="center", fontsize=FS["label"], color=T_TITLE, weight="bold", transform=trf, clip_on=False)
    save(p, "ex2_three_input_types")


# ---------------------------------------------------------------- ex3: name one object, classify another, by depth alone (D2)
D2_ARMS = [
    ("receiver alone", "not patched", {}, ""),
    ("donor early only", "train patched into layers input to 10", {"X": range(-1, 11)}, ""),
    ("donor early, receiver restored in the band", "train patched early, cat restored in 11 to 22", {"X": range(-1, 11), "Z": range(11, 23)}, "names the cat, says it is not an animal"),
    ("donor in the band only", "train patched into layers 11 to 22", {"X": range(11, 23)}, "names the train, says it is an animal"),
    ("donor's early-window state in place (control)", "train's layers 7 to 10, patched at 7 to 10", {"X": range(7, 11)}, ""),
    ("donor's band state delivered in the early window", "train's layers 11 to 14, back-patched to 7 to 10", {"X": range(7, 11)}, "back-patched, its later state is used"),
    ("donor's early state delivered in the band", "train's layers 7 to 10, patched at 11 to 14", {"X": range(11, 15)}, "patched later, its earlier state is ignored"),
]


def ex3():
    d = pd.read_csv(run_file("back_patching/Qwen3.5-4B/back_patching.csv"))
    d = d[(d.scene == 1) & (d.source == "picture")]
    assert set(d.X) == {"train"} and set(d.Z) == {"cat"}
    style()
    p = Page(7.1)
    header(
        p,
        "Patched early, the train is used; patched into the middle layers, it is only named",
        "The train run's states are patched into the cat run at the dark layers and the cat's own are restored at the mid-grey ones  \u00b7  Qwen3.5-4B",
    )
    labels(p, 0.0, 0.85, KEY("train", "cat"))
    tx, tw = 4.35, W_IN - 4.35
    nr, xmax = len(D2_ARMS), 5.25
    unit = tw / xmax  # inches per table unit
    strip(
        p,
        tx + 3.2 / 2 * unit,
        0.75,
        [("img", slot(CANVAS["scene1_donor_train"], 5), "donor sees"), ("arrow", "residual stream\npatched in", 1.4), ("img", slot(CANVAS["scene1_receiver_cat"], 5), "receiver sees")],
    )
    ax = p.axes(tx, 2.75, tw, 4.0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(nr, 0)
    ax.axis("off")
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    bw = 1.16 / 33
    ax.text(0.02, -0.14, "train patched", ha="left", va="bottom", fontsize=FS["col"], color=G(0.25))
    ax.text(0.66, -0.14, "cat restored", ha="left", va="bottom", fontsize=FS["col"], color=G(0.55))
    ax.text(1.75, -0.14, "What is it?", ha="center", va="bottom", fontsize=FS["col"], color=T_SEC)
    ax.text(2.7, -0.14, "Is it an animal?", ha="center", va="bottom", fontsize=FS["col"], color=T_SEC)
    for i, (arm, label, spec, note) in enumerate(D2_ARMS):
        if i == 4:
            ax.plot([0.02, 3.2], [i, i], color=G(0.8), lw=0.8, clip_on=False)
            ax.text(3.32, i, "cross-layer patching: the same states at another depth", ha="left", va="center", fontsize=FS["col"], color=T_SEC, clip_on=False)
        for k, L in enumerate(range(-1, 32)):
            c = G(0.25) if L in spec.get("X", ()) else G(0.62) if L in spec.get("Z", ()) else G(0.9)
            ax.add_patch(Rectangle((0.02 + k * bw, i + 0.3), bw * 0.82, 0.4, facecolor=c, edgecolor="none"))
        ax.text(-0.01, i + 0.5, label, ha="right", va="center", fontsize=FS["label"], color=T_LABEL, transform=tr, clip_on=False)
        for task, x0 in [("say", 1.3), ("animal", 2.25)]:
            r = one(d, task=task, arm=arm)
            k_ = kind_of(r)
            ax.add_patch(Rectangle((x0, i + 0.09), 0.9, 0.82, facecolor=fill(k_, FAM[task]), edgecolor="none"))
            ax.text(x0 + 0.45, i + 0.5, shown(task, r.generation), ha="center", va="center", fontsize=FS["cell"], color=ink(k_))
            log(
                "ex3",
                "Qwen3.5-4B",
                "COCO photo in slot 5 of a six-slot canvas",
                "inputs/scene1_receiver_cat.png; donor inputs/scene1_donor_train.png",
                f"{INTRO} [image] " + Q[task].format(q=5),
                label.replace("train's", "the train run's").replace("cat restored", "the cat's own states restored"),
                "runs/back_patching/Qwen3.5-4B/back_patching.csv",
                f"source=picture, scene=1, task={task}, arm={arm}",
                task,
                r.generation,
                k_,
            )
        if note:
            ax.text(3.32, i + 0.5, note, ha="left", va="center", fontsize=FS["key"], color=T_LABEL)
    for L in (-1, 11, 23, 31):
        ax.text(0.02 + (L + 1) * bw + bw * 0.41, nr + 0.06, "input" if L == -1 else str(L), ha="center", va="top", fontsize=10, color=T_SEC)
    ax.text(0.6, nr + 0.42, "layer", ha="center", va="top", fontsize=10, color=T_SEC)
    save(p, "ex3_name_one_classify_another")


# ---------------------------------------------------------------- ex4: one answer, two objects (F4)
F4_ARMS = [
    ("receiver alone", "not patched", None),
    ("full state, read layers", "all 2,560 dimensions", None),
    ("lens part, read layers", "the 256 dimensions the lens reads", None),
    ("complement of the lens part, read layers", "the other 2,304 dimensions", None),
    ("random part, read layers, draw 1", "a random 256 dimensions (3 tries)", ["random part, read layers, draw 2", "random part, read layers, draw 3"]),
    ("placebo donor (empty canvas), read layers", "all of it, from an empty canvas", None),
]


def ex4():
    d = pd.read_csv(run_file("subspace_patching/Qwen3.5-4B/subspace_patching.csv"))
    d = d[d.scene == 1]
    assert set(d.X) == {"train"} and set(d.Z) == {"cat"}
    truth = {"say": ("train", "cat"), "animal": ("3", "7")}
    text, kinds, fams, two = [], [], [], {}
    for i, (arm, lab, same) in enumerate(F4_ARMS):
        tru = {t: ("\x00", z) for t, (_, z) in truth.items()} if arm.startswith("placebo") else truth  # the placebo donor saw no object
        row, krow = [], []
        for t in ["say", "animal"]:
            g = one(d, task=t, arm=arm).generation
            for other in same or []:
                assert str(one(d, task=t, arm=other).generation).strip() == str(g).strip(), "the random draws differ"
            row.append(shown(t, g))
            krow.append(whose(g, *tru[t]))
        g = str(one(d, task="animal then name", arm=arm).generation).strip()
        for other in same or []:
            assert str(one(d, task="animal then name", arm=other).generation).strip() == g
        digit, noun = g[:1], g[1:].strip(" .,").lower()
        two[(i, 2)] = (shown("animal", digit) + ",", whose(digit, *tru["animal"]), "rule"), (noun, whose(noun, *tru["say"]), "name")
        row.append(g)
        krow.append("N")
        text.append(row)
        kinds.append(krow)
        fams.append(["name", "rule", "rule"])
        for t in ["say", "animal", "animal then name"]:
            for a in [arm] + (same or []):
                r = one(d, task=t, arm=a)
                log(
                    "ex4",
                    "Qwen3.5-4B",
                    "COCO photo in slot 5 of a six-slot canvas",
                    "inputs/scene1_receiver_cat.png; donor inputs/scene1_donor_train.png",
                    f"{INTRO} [image] " + Q[t].format(q=5),
                    lab + (", patched at layers 9, 11, ..., 29" if i else ""),
                    "runs/subspace_patching/Qwen3.5-4B/subspace_patching.csv",
                    f"scene=1, task={t}, arm={a}",
                    t,
                    r.generation,
                    whose(r.generation, *tru[t]) if t != "animal then name" else "two parts, see the figure",
                )
    style()
    p = Page(6.6)
    header(p, "One answer can name the train and treat it as the cat", "Subspace patching: part of the train run's residual stream is patched into the cat run at layers 9 to 29  \u00b7  Qwen3.5-4B")
    labels(p, 0.0, 0.85, KEY("train", "cat"))
    tx, tw = 3.75, W_IN - 3.75
    strip(
        p,
        tx + tw / 2,
        0.75,
        [("img", slot(CANVAS["scene1_donor_train"], 5), "donor sees"), ("arrow", "part of the\nresidual stream", 1.5), ("img", slot(CANVAS["scene1_receiver_cat"], 5), "receiver sees")],
    )
    grid(
        p.axes(tx, 3.0, tw, 3.5),
        text,
        kinds,
        fams,
        [lab for _, lab, _ in F4_ARMS],
        ["What is it?", "Is it an animal?", "One question:\n'Is it an animal, and what is it?'"],
        two=two,
        col_note="what was patched",
    )
    save(p, "ex4_one_answer_two_objects")


# ---------------------------------------------------------------- ex5: a name added to the picture (G1)
G1_DOG = [
    ("own picture", "clean", 0.0, "no steering"),
    ("inject at the all image tokens", "other name", 4.0, "steer toward 'car'"),
    ("inject at the all image tokens", "random direction", 4.0, "steer toward a random direction"),
    ("inject at the all image tokens", "other name, W_U only", 4.0, "steer toward 'car' without the lens"),
]
G1_EMPTY = [
    ("blank canvas", "none", 0.0, "no steering"),
    ("blank canvas", "own name", 4.0, "steer toward 'dog'"),
    ("blank canvas", "random direction, blank", 8.0, "steer toward a random direction, 2x"),
]


def ex5():
    d = pd.read_csv(run_file("name_steering/Qwen3.5-4B/name_steering.csv"))
    d = d[d.scene == 0]
    assert set(d.X) == {"dog"} and set(d.Y) == {"car"}
    truth_dog = {"say": ("car", "dog"), "letter": ("C", "D"), "animal": ("3", "7"), "ride": ("7", "3")}
    truth_empty = {"say": ("dog", "\x00"), "letter": ("D", "\x00"), "animal": ("7", "\x00"), "ride": ("\x00", "\x00")}  # "3" is also the no-object answer
    tasks = ["say", "letter", "animal", "ride"]

    def rows(spec, truth, img, data):
        text, kinds, fams = [], [], []
        for block, arm, a, lab in spec:
            r_, k_ = [], []
            for t in tasks:
                g = one(d, block=block, arm=arm, alpha=a, task=t).generation
                r_.append(shown(t, g))
                k_.append(whose(g, *truth[t]))
                log(
                    "ex5",
                    "Qwen3.5-4B",
                    data,
                    img,
                    f"{INTRO} [image] " + Q[t].format(q=3),
                    lab + ("" if lab.startswith("no ") else (", every image token, layers 9, 11, ..., 29" if block != "blank canvas" else ", slot 3's 49 tokens, layers 9, 11, ..., 29")),
                    "runs/name_steering/Qwen3.5-4B/name_steering.csv",
                    f"scene=0, block={block}, arm={arm}, alpha={a}, task={t}",
                    t,
                    g,
                    whose(g, *truth[t]),
                )
            text.append(r_)
            kinds.append(k_)
            fams.append([FAM[t] for t in tasks])
        return text, kinds, fams

    t1, k1, f1 = rows(G1_DOG, truth_dog, "inputs/g1_scene0_dog.png", "COCO photo in slot 3 of a six-slot canvas")
    t2, k2, f2 = rows(G1_EMPTY, truth_empty, "inputs/empty_canvas.png", "the empty six-slot canvas")
    style()
    p = Page(7.45)
    header(
        p,
        "Steer a dog picture toward 'car': the model says car, but still treats it as a dog",
        "Activation steering: one word's J-lens direction is added to the picture's residual stream at layers 9 to 29, at one strength  \u00b7  Qwen3.5-4B",
    )
    labels(p, 0.0, 0.55, [(BLUE, "blue: names the steering word"), (G(0.45), "grey: answers for what it sees")], across=3.0)
    cols = ["What is it?", "First letter?", "Is it an animal?", "Is it ridden\nor driven?"]
    tx, tw, rh = 5.0, W_IN - 5.0, 0.56
    panel(p, 1.1, "a", "A dog")
    show(p.axes(0.0, 1.95 + (4 * rh - 1.6) / 2, 1.6, 1.6), slot(CANVAS["g1_scene0_dog"], 3))
    grid(p.axes(tx, 1.95, tw, 4 * rh), t1, k1, f1, [lab for *_, lab in G1_DOG], cols)
    panel(p, 4.75, "b", "An empty slot")
    show(p.axes(0.0, 5.6 + (3 * rh - 1.6) / 2, 1.6, 1.6), slot(CANVAS["empty"], 3))
    grid(p.axes(tx, 5.6, tw, 3 * rh), t2, k2, f2, [lab for *_, lab in G1_EMPTY], cols)
    save(p, "ex5_written_name")


# ---------------------------------------------------------------- photo columns (ex6, ex8): a picture over a column of answer boxes
def photo_columns(p, pics, rows, row_labels, top_img, img_h, top_rows, x0=2.55, col_w=2.55, gap=0.17, box_h=0.5, step=0.6, hi=F_BLUE):
    """pics[k] = (PIL, label or None, caption or None); rows[k][j] = (text, highlighted)"""
    for k, (im, lab, cap) in enumerate(pics):
        x = x0 + k * (col_w + gap)
        show(p.axes(x, top_img, col_w, img_h), im, lab, anchor="S", label_loc="center")
        if cap:
            p.fig.text(p.x(x + col_w / 2), p.y(top_img + img_h + 0.12), cap, fontsize=FS["key"], color=T_TITLE, ha="center", va="top", linespacing=1.25)
        for j, (txt, on) in enumerate(rows[k]):
            top = top_rows + j * step
            p.fig.patches.append(Rectangle((p.x(x), p.y(top + box_h)), p.x(col_w), box_h / p.h, transform=p.fig.transFigure, facecolor=hi if on else F_OWN, edgecolor="none"))
            p.fig.text(p.x(x + col_w / 2), p.y(top + box_h / 2), txt, fontsize=FS["cell"], color=T_TITLE if on else G(0.35), va="center", ha="center")
    for j, lab in enumerate(row_labels):
        p.fig.text(p.x(x0 - 0.15), p.y(top_rows + j * step + box_h / 2), lab, fontsize=FS["label"], color=T_LABEL, va="center", ha="right", linespacing=1.15)


# ---------------------------------------------------------------- ex6: GQA photos, the answer word's single lens direction removed (N1)
GQA = [("n508641", "Who is catching the round ball?"), ("n296467", "What dessert is rainbow colored?"), ("n204894", "What's the child doing?"), ("n130638", "What's the baseball bat made of?")]
GQA_ARMS = [("clean", "its answer"), ("ablate target word's lens direction", "its answer's\ndirection ablated"), ("ablate another word's lens direction", "another word's\ndirection ablated")]


def ex6():
    n = pd.read_csv(run_file("gqa_ablation/Qwen3.5-4B/gqa_ablation.csv"))
    imtab = pd.read_parquet(f"{DATA}/gqa/testdev_balanced_images/testdev-00000-of-00001.parquet")
    imtab = imtab[imtab.id.isin([i for i, _ in GQA])].set_index("id")
    pics, rows = [], []
    for iid, q in GQA:
        im = stimuli.fit_photo(imtab.loc[iid, "image"]["bytes"])
        IMG[f"gqa_{iid}"] = im
        sub = n[(n.imageId == iid) & (n.question == q)]
        clean = str(one(sub, arm="clean").generation).strip()
        col = []
        for j, (arm, lab) in enumerate(GQA_ARMS):
            r = one(sub, arm=arm)
            g = str(r.generation).strip()
            col.append((g, j > 0 and g.lower() != clean.lower()))
            log(
                "ex6",
                "Qwen3.5-4B",
                "GQA testdev-balanced photo",
                f"inputs/gqa_{iid}.png",
                f"[image] {q} Answer with one word.",
                lab.replace("\n", " ") if j else "no ablation",
                "runs/gqa_ablation/Qwen3.5-4B/gqa_ablation.csv",
                f"imageId={iid}, question={q!r}, arm={arm}; reference={r.reference}",
                "gqa",
                g,
                "correct" if truthy(r.correct) else "wrong",
            )
        pics.append((im, None, textwrap.fill(q, 27)))
        rows.append(col)
    style()
    p = Page(6.75)
    header(
        p,
        "On real photos, ablating the direction of the model's answer word changes its answer",
        "Directional ablation: one word's J-lens direction is removed from the photo's residual stream at layers 9 to 29  \u00b7  Qwen3.5-4B, GQA",
    )
    labels(p, 0.0, 0.55, [(BLUE, "blue: the answer changed"), (G(0.45), "13 of the 91 answers it got right changed; ablating another word's direction changed none")], across=3.0)
    photo_columns(p, pics, rows, [lab for _, lab in GQA_ARMS], top_img=0.95, img_h=2.75, top_rows=4.6)
    save(p, "ex6_gqa_one_direction")


# ---------------------------------------------------------------- ex7: a written word, and the one direction that decides it (A1, A2)
def ex7():
    p1, p2 = "printed_word/Qwen3.5-4B/printed_word.csv", "scam_notes/Qwen3.5-4B/scam_notes.csv"
    a1 = pd.read_csv(run_file(p1))
    a1 = a1[a1.scene == 5]
    assert set(a1.X) == {"clock"} and set(a1.Y) == {"boat"}
    a2 = pd.read_csv(run_file(p2))
    a2 = a2[a2.item == "img_0706"]
    assert set(a2.X) == {"sausage"} and set(a2.Y) == {"helmet"}
    IMG["a1_scene5_clock_printed_boat"] = stimuli.sheet_frame(run_file("printed_word/Qwen3.5-4B/stimuli.png"), 5, 3, 224, 224)
    IMG["scam_img_0706"], IMG["scam_img_0706_note_blanked"] = scam_photo("SCAM_img_0706.jpg"), scam_photo("NoSCAM_img_0706.jpg")
    A1_ROWS = [
        (("arm", "clean"), "no ablation"),
        (("arm", "ablate label word Y"), "ablate 'boat'"),
        (("arm", "ablate random direction"), "ablate a random direction"),
        (("arm", "ablate other word Z"), "ablate another word"),
        (("arm", "ablate object word X"), "ablate 'clock'"),
    ]
    A2_ROWS = [
        (("none", 0.0), "no ablation"),
        (("ablate note word Y", 0.0), "ablate 'helmet'"),
        (("ablate random direction", 0.0), "ablate a random direction"),
        (("ablate unrelated word", 0.0), "ablate another word"),
        (("ablate object word X", 0.0), "ablate 'sausage'"),
    ]
    TW_ROWS = [
        (("none", 0.0), "no steering"),
        (("note word Y", 2.0), "steer toward 'helmet', weak"),
        (("note word Y", 4.0), "steer toward 'helmet', strong"),
        (("random, matched norm", 4.0), "steer toward random, strong"),
    ]

    def rows(df, spec, kind, img, data, path):
        text, kinds, fams, clean = [], [], [], None
        for keyv, lab in spec:
            r = one(df, arm=keyv[1]) if keyv[0] == "arm" else one(df, kind=kind, arm=keyv[0], alpha=keyv[1])
            g = str(r.generation).strip()
            clean = g if clean is None else clean
            changed = g.lower() != clean.lower()
            text.append([g])
            kinds.append(["X" if changed else "Z"])
            fams.append(["name"])
            log(
                "ex7",
                "Qwen3.5-4B",
                data,
                img,
                "[image] What object is this? Reply with the noun only.",
                lab + ("" if lab.startswith("no ") else ", every image token, layers 9, 11, ..., 29"),
                f"runs/{path}",
                (f"scene=5, arm={keyv[1]}" if keyv[0] == "arm" else f"item=img_0706, kind={kind}, arm={keyv[0]}, alpha={keyv[1]}"),
                "name",
                g,
                "changed" if changed else "unchanged",
            )
        return text, kinds, fams

    style()
    p = Page(7.5)
    header(
        p,
        "A written word can beat the object, and ablating that word's direction brings the object back",
        "'What object is this?'  \u00b7  directional ablation (a, b) and activation steering (c) with one word's J-lens direction, every image token, layers 9 to 29  \u00b7  Qwen3.5-4B",
    )
    labels(p, 0.0, 0.55, [(BLUE, "blue: the answer changed")])
    cols = [
        (IMG["a1_scene5_clock_printed_boat"], "a", "A clock with 'boat' printed on it", a1, A1_ROWS, None, "inputs/a1_scene5_clock_printed_boat.png", "COCO crop with a printed word", p1),
        (IMG["scam_img_0706"], "b", "A photo with a note saying 'Helmet'", a2, A2_ROWS, "ablation", "inputs/scam_img_0706.png", "SCAM photo with a hand-written note", p2),
        (IMG["scam_img_0706_note_blanked"], "c", "The same photo, note left blank", a2, TW_ROWS, "injection", "inputs/scam_img_0706_note_blanked.png", "NoSCAM twin (note blanked)", p2),
    ]
    col_w, gap, lab_w, ans_w, rh = 4.2, 0.27, 2.75, 1.4, 0.5
    for c, (im, letter, title, df, spec, kind, img, data, path) in enumerate(cols):
        x0 = c * (col_w + gap)
        p.fig.text(p.x(x0), p.y(1.0), letter, fontsize=FS["panel"], color=T_TITLE, weight="bold", ha="left", va="top")
        p.fig.text(p.x(x0 + 0.24), p.y(1.0), title, fontsize=FS["panel"], color=T_TITLE, ha="left", va="top")
        show(p.axes(x0, 1.45, col_w, 2.6), im, anchor="C")
        t, k, f = rows(df, spec, kind, img, data, path)
        grid(p.axes(x0 + lab_w, 4.35, ans_w, rh * len(spec)), t, k, f, [lab for _, lab in spec], [""])
    save(p, "ex7_written_word")


# ---------------------------------------------------------------- ex8: what the model says about where its answer came from (A1, A2)
def ex8():
    it = pd.read_csv(run_file("scam_notes/Qwen3.5-4B/items.csv")).set_index("key")
    a2 = pd.read_csv(run_file("scam_notes/Qwen3.5-4B/scam_notes.csv"))
    sc = pd.read_csv(run_file("printed_word/Qwen3.5-4B/scenes.csv")).set_index("scene")
    a1 = pd.read_csv(run_file("printed_word/Qwen3.5-4B/printed_word.csv"))
    A1S = run_file("printed_word/Qwen3.5-4B/stimuli.png")
    Q2_A2 = "Did your answer come from the object you see or from writing in the image? Answer object or writing."
    Q2_A1 = "Did your answer come from the object shown in the picture or from the printed word? Reply with one word: object or word."
    pics, rows = [], []
    for key_, lab in [("img_1009", "a speaker, note 'Tree'"), ("img_0650", "a box, note 'Lion'")]:
        r = it.loc[key_]
        g = str(one(a2, kind="ablation", arm="none", item=key_).generation).strip()
        IMG[f"scam_{key_}"] = scam_photo(r.file_name)
        did = "the object" if truthy(r.clean_says_X) else "the writing"
        ok = (r.source_report == "object") == truthy(r.clean_says_X)
        pics.append((IMG[f"scam_{key_}"], lab, None))
        rows.append([(g, False), (r.source_report, not ok), (did, False)])
        log(
            "ex8",
            "Qwen3.5-4B",
            "SCAM photo with a hand-written note",
            f"inputs/scam_{key_}.png",
            "[image] What object is this? Reply with the noun only.",
            "none (turn 1)",
            "runs/scam_notes/Qwen3.5-4B/items.csv",
            f"key={key_}",
            "name",
            g,
            did,
        )
        log(
            "ex8",
            "Qwen3.5-4B",
            "SCAM photo with a hand-written note",
            f"inputs/scam_{key_}.png",
            f"turn 2 after its own answer: {Q2_A2}",
            "none (turn 2)",
            "runs/scam_notes/Qwen3.5-4B/items.csv",
            f"key={key_}",
            "source report (first word)",
            r.source_report,
            "matches" if ok else "does not match",
        )
    for s in [4, 5]:
        r = sc.loc[s]
        g = str(one(a1, arm="clean", scene=s).generation).strip()
        name = f"a1_scene{s}_{r.X}_printed_{r.Y}"
        IMG[name] = stimuli.sheet_frame(A1S, s, 3, 224, 224)
        did = "the object" if truthy(r.clean_says_X) else "the word"
        ok = (r.source_report == "object") == truthy(r.clean_says_X)
        pics.append((IMG[name], f"{'an' if r.X[0] in 'aeiou' else 'a'} {r.X}, '{r.Y}' printed", None))
        rows.append([(g, False), (r.source_report, not ok), (did, False)])
        log(
            "ex8",
            "Qwen3.5-4B",
            "COCO crop with a printed word",
            f"inputs/{name}.png",
            "[image] What object is this? Reply with the noun only.",
            "none (turn 1)",
            "runs/printed_word/Qwen3.5-4B/scenes.csv",
            f"scene={s}",
            "name",
            g,
            did,
        )
        log(
            "ex8",
            "Qwen3.5-4B",
            "COCO crop with a printed word",
            f"inputs/{name}.png",
            f"turn 2 after its own answer: {Q2_A1}",
            "none (turn 2)",
            "runs/printed_word/Qwen3.5-4B/scenes.csv",
            f"scene={s}",
            "source report (first word)",
            r.source_report,
            "matches" if ok else "does not match",
        )
    style()
    p = Page(6.6)
    header(
        p,
        "Asked where its answer came from, the model says 'the writing' even when it named the object",
        "We ask 'What object is this?', then 'Did your answer come from the object you see or from writing in the image?'  \u00b7  Qwen3.5-4B",
    )
    labels(p, 0.0, 0.55, [(ORANGE, "orange: its explanation is wrong")])
    photo_columns(p, pics, rows, ["its answer", "it says the answer\ncame from", "what it really named"], top_img=1.3, img_h=2.75, top_rows=4.45, hi=F_ORANGE)
    save(p, "ex8_source_report")


# ---------------------------------------------------------------- the inputs at full size, small copies for REPORT.md, and outputs.csv
def assets():
    os.makedirs(f"{OUT}/inputs", exist_ok=True)
    for name, im in CANVAS.items():
        im.save(f"{OUT}/inputs/{name}.png")
    for name, s in [
        ("scene0_donor_bird", 1),
        ("scene0_receiver_airplane", 1),
        ("scene1_donor_train", 5),
        ("scene1_receiver_cat", 5),
        ("scene1_donor_printed_train", 5),
        ("g1_scene0_dog", 3),
        ("empty", 3),
    ]:
        slot(CANVAS[name], s).save(f"{OUT}/inputs/{name}_slot{s}.png")
    for name, im in IMG.items():
        im.save(f"{OUT}/inputs/{name}.png")
    pd.DataFrame(ROWS).to_csv(f"{OUT}/outputs.csv", index=False)
    print("wrote", f"{OUT}/inputs and outputs.csv ({len(ROWS)} rows)")


def main():
    global RUNS, DATA, OUT, FALLBACK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR, help="the run folders to read (default runs/; results/ for the original runs)")
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None, help="default <runs>/examples")
    ap.add_argument("--fallback", type=Path, default=None, help="read runs missing from --runs here (e.g. results)")
    args = ap.parse_args()
    RUNS, DATA = str(args.runs), str(args.data)
    FALLBACK = str(args.fallback) if args.fallback else None
    OUT = str(args.out or args.runs / "examples")
    shutil.rmtree(f"{OUT}/inputs", ignore_errors=True)
    canvases()
    for f in (ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, assets):
        f()


if __name__ == "__main__":
    main()
