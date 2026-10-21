#!/usr/bin/env python
"""Worked examples for the 21 October results (RECORD.md rows 97 to 133), in the format of results/examples/ex1 to ex8: the exact
inputs a model received and its answers as logged, real scenes, nothing re-run. Each cell also carries, in small type, how many of
the run's 64 scenes gave that cell's outcome. The canvases are rebuilt from each run's crops.json with the run's own stimulus code
(data/coco2017), or, when COCO is absent, from the tiles saved in results/samples; every answer is read from the run's CSV in
results/oct21.
    .venv/bin/python scripts/fig_examples21.py      # writes results/examples/ex9 to ex14 (.png, .pdf, .svg), inputs/, outputs_21oct.csv,
                                                    # and results/samples (every tile of the two crop banks, every receiver canvas)
"""

import json
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    """a run's CSV, stored as .csv or, in results/, as .csv.gz"""
    p = Path(path)
    return pd.read_csv(p if p.exists() else p.with_name(p.name + ".gz"))


for d in ("jbound", "scripts", "experiments", "src"):
    sys.path.insert(0, str(ROOT / d))
import fig_examples as E  # noqa: E402  the 14 October helpers: Page, header, labels, strip, grid, save, log
from cross_modal_patch import scenes  # noqa: E402
from fig_battery import rebuild_bank  # noqa: E402
from read_battery import QUESTIONS  # noqa: E402
from source_report import QUESTION as SOURCE_Q  # noqa: E402

from jspace import stimuli  # noqa: E402

E.OUT = str(ROOT / "results" / "examples")
RES = ROOT / "results" / "oct21"
SAMPLES = ROOT / "results" / "samples"
COCO = ROOT / "data" / "coco2017"
INTRO = E.INTRO
N = 64
CANVAS = {}  # name -> PIL canvas, saved to inputs/
BANKS = {}  # model -> (words, bank), for results/samples


def table(run, model):
    """the run's 64 scenes (X, Z, q, tiles) from its crops.json, with the run's own scene sampler"""
    C = json.load(open(RES / run / model / "crops.json"))
    words = C["vocab"]
    if (COCO / "val2017.zip").exists():
        w2, bank = rebuild_bank(COCO, RES / run / model / "crops.json")
        assert w2 == words, (run, model)
    else:  # without COCO: the tiles saved by a previous build (results/samples/<model>/bank)
        bank = {}
        for f in sorted((SAMPLES / model / "bank").glob("*.png")):
            w, _ = f.stem.rsplit("_", 1)
            bank.setdefault(w, []).append(Image.open(f).convert("RGB"))
    assert set(words) <= set(bank), (run, model)  # the bank keeps the words the gate dropped (under four tiles)
    BANKS.setdefault(model, (words, bank))
    return scenes(words, bank, N), bank


def foot(p, tx, top, text):
    """the note under a grid, wrapped to the grid's width"""
    p.fig.text(p.x(tx), p.y(top), textwrap.fill(text, 150), fontsize=E.FS["col"], color=E.T_SEC, ha="left", va="top", linespacing=1.3)


def canvas(name, q, tile):
    CANVAS[name] = stimuli.scene(q, tile)
    return CANVAS[name]


def article(w):
    return "an" if w[0] in "aeiou" else "a"


def qt(t):
    """the question template of read_battery.QUESTIONS[t] (a tuple with its kind and answer type)"""
    v = QUESTIONS[t]
    return v[0] if isinstance(v, (tuple, list)) else v


def short(prompt, reference):
    """'How many spoons are there in the image?' -> 'spoons (6)'"""
    m = re.search(r"How many (.+?) are there", prompt)
    return f"{m.group(1) if m else prompt.split(chr(10))[0]} ({reference})"


def rel(run, model, f):
    return f"results/oct21/{run}/{model}/{f}"


def split(gen):
    """'cup, seen' -> ('cup', 'seen'); anything else kept whole"""
    g = str(gen).strip().lower().rstrip(".")
    if "," in g:
        a, b = g.split(",", 1)
        return a.strip(), b.strip()
    return g, ""


def kinds_of(gen, read_objects):
    """the two outcome kinds of a 'name, channel' answer: the name blue when it is an object that was read, the channel blue when 'read'"""
    nm, ch = split(gen)
    return ("X" if nm in read_objects else "Z", "X" if ch == "read" else "Z")


def pair(gen, read_objects):
    nm, ch = split(gen)
    kn, kc = kinds_of(gen, read_objects)
    return ((nm, kn, "name"), (ch, kc, "name"))


def marks(ax, counts):
    """the small count in every cell: of the run's 64 scenes, how many gave this cell's outcome"""
    for (i, j), n in counts.items():
        n, tot = n if isinstance(n, tuple) else (n, N)
        ax.text(j + 0.95, i + 0.885, f"{n}/{tot}", ha="right", va="bottom", fontsize=6.8, color=E.G(0.5))


def two_grid(ax, cells, row_labels, col_labels, counts=None, fs=11, **kw):
    """a grid where every cell is a (left, right) pair of boxes"""
    nr, nc = len(cells), len(cells[0])
    two = {(i, j): cells[i][j] for i in range(nr) for j in range(nc)}
    E.grid(ax, [[""] * nc for _ in range(nr)], [["Z"] * nc for _ in range(nr)], [["name"] * nc for _ in range(nr)], row_labels, col_labels, fs=fs, two=two, **kw)
    if counts:
        marks(ax, counts)


COUNT_NOTE = "Small numbers: of the 64 scenes, how many gave that cell's outcome."


# ---------------------------------------------------------------- ex9: the word's read written into the picture's run (R9, row 101)
def ex9():
    run, model = "run13", "Qwen3.5-2B"
    S, bank = table(run, model)
    r = S.iloc[0]
    assert (r.X, r.Z, r.q) == ("bird", "airplane", 1)
    D = read_csv(RES / run / model / "cross_modal_patch.csv")
    d = D[D.scene == 0]
    cz = canvas("scene0_2B_receiver_airplane", r.q, r.tz)
    canvas("scene0_2B_donor_bird", r.q, r.tx)
    arms = [
        ("clean", "not\npatched"),
        ("head 5 of 19", "head 5\nof 19"),
        ("all heads of 7", "all of 7"),
        ("all heads of 11", "all of 11"),
        ("all heads of 15", "all of 15"),
        ("all heads of 19", "all of 19"),
        ("all heads of 7+11", "7 and 11"),
        ("all heads of 11+15+19", "11, 15\nand 19"),
        ("all heads of 7+11+15+19", "all four"),
    ]
    tasks = ["say", "animal"]
    tables = {}
    for direction, donor_in, rec_in in [("typed>picture", "none (text only)", "inputs/scene0_2B_receiver_airplane.png"), ("picture>typed", "inputs/scene0_2B_donor_bird.png", "none (text only)")]:
        text, kinds, fams, counts = [], [], [], {}
        for i, t in enumerate(tasks):
            rows = [E.one(d, task=t, direction=direction, arm=a) for a, _ in arms]
            text.append([E.shown(t, x.generation) for x in rows])
            kinds.append([E.kind_of(x) for x in rows])
            fams.append([E.FAM[t]] * len(rows))
            for j, (_x, (a, _)) in enumerate(zip(rows, arms)):
                g = D[(D.task == t) & (D.direction == direction) & (D.arm == a)]
                counts[(i, j)] = int(sum(E.kind_of(y) == kinds[i][j] for y in g.itertuples()))
            q = qt(t).format(q=1, a="", b="", c="", d="", e="")
            for x, (a, _) in zip(rows, arms):
                if direction == "typed>picture":
                    prompt = f"{INTRO} [image] {q}"
                    iv = "not patched" if a == "clean" else f"the head outputs recorded while reading '{INTRO} Image slot 1 contains a bird. {q}' written in at the question's tokens: {a}"
                else:
                    prompt = f"{INTRO} Image slot 1 contains an airplane. {q}"
                    iv = "not patched" if a == "clean" else f"the head outputs recorded while seeing the bird written in at the question's tokens: {a}"
                E.log(
                    "ex9",
                    model,
                    "COCO photo in slot 1 of a six-slot canvas, or the sentence",
                    rec_in + ("; donor " + donor_in if a != "clean" else ""),
                    prompt,
                    iv,
                    rel(run, model, "cross_modal_patch.csv"),
                    f"scene=0, task={t}, direction={direction}, arm={a}",
                    t,
                    x.generation,
                    E.kind_of(x),
                )
        tables[direction] = (text, kinds, fams, counts)
    E.style()
    p = E.Page(8.0)
    E.header(
        p,
        "The heads that read a typed word, written into the run that sees a picture, carry the word's object",
        f"Head-output patching at the question's tokens: the reader heads' outputs recorded while reading 'Image slot 1 contains a bird.' replace the airplane run's  ·  {model}",
    )
    E.labels(p, 0.0, 0.85, E.KEY("bird", "airplane"))
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(p, tx + tw / 2, 0.75, [("text", "'Image slot 1\ncontains a bird.'", "donor reads", 1.75), ("arrow", "reader heads'\noutputs written in", 1.5), ("img", E.slot(cz, 1), "receiver sees")])
    E.panel(p, 2.55, "a", "the word's read, written into the picture's run")
    text, kinds, fams, counts = tables["typed>picture"]
    ax = p.axes(tx, 3.3, tw, 1.1)
    E.grid(ax, text, kinds, fams, [E.LABEL[t] for t in tasks], [c for _, c in arms], col_note="heads written in")
    marks(ax, counts)
    E.panel(p, 4.9, "b", "the reverse: the picture's read, written into the run that reads 'Image slot 1 contains an airplane.'")
    text, kinds, fams, counts = tables["picture>typed"]
    ax = p.axes(tx, 5.65, tw, 1.1)
    E.grid(ax, text, kinds, fams, [E.LABEL[t] for t in tasks], [c for _, c in arms], col_note="heads written in")
    marks(ax, counts)
    foot(
        p,
        tx,
        7.2,
        f"Scene 0 of 64. {COUNT_NOTE} The word's read moves the animal rule on 63 of 64 scenes with the heads of 7 and 11 and the name "
        "on 61 of 64 with 11, 15 and 19; the picture's read moves the sentence run's rule on 61 and its name on 26 (row 101).",
    )
    E.save(p, "ex9_word_read_into_picture")


# ---------------------------------------------------------------- ex10: does the model say it saw what it read? (R10, rows 103, 105)
def ex10():
    run, model = "run15", "Qwen3.5-4B"
    S, bank = table(run, model)
    D = read_csv(RES / run / model / "source_report.csv")
    D = D[D.target == "q"]
    B = read_csv(RES / "run15b" / model / "source_report.csv")
    B = B[B.target == "q"]
    arms = [
        (D, "clean", "not patched"),
        (D, "donor", "the donor alone"),
        (B, "all heads of 19+23", "heads of 19 and 23"),
        (D, "all heads of 15+19+23", "15, 19 and 23"),
        (D, "all heads of 11+15+19+23", "11 to 23"),
        (D, "all heads of 7+11+15+19+23", "7 to 23"),
    ]
    cells, counts, labs = [], {}, []
    for i, s in enumerate((0, 1)):
        r = S.iloc[s]
        row_d, row_b = D[D.scene == s], B[B.scene == s]
        W, pslot = row_d.W.iloc[0], int(row_d.p.iloc[0])
        name = f"scene{s}_4B_receiver_{r.Z}"
        cz = canvas(name, r.q, r.tz)
        note_r = f"A note says: slot {pslot} contains {article(W)} {W}."
        note_d = f"A note says: slot {r.q} contains {article(r.X)} {r.X}. {note_r}"
        read = {r.X, W}
        row = []
        for j, (src, a, _) in enumerate(arms):
            x = E.one(row_d if src is D else row_b, arm=a)
            row.append(pair(x.generation, read))
            g = src[src.arm == a]
            counts[(i, j)] = int(sum(kinds_of(y.generation, {y.X, y.W}) == kinds_of(x.generation, read) for y in g.itertuples()))
            prompt = f"{INTRO} {note_d} {SOURCE_Q.format(s=r.q)}" if a == "donor" else f"{INTRO} [image] {note_r} {SOURCE_Q.format(s=r.q)}"
            iv = "not patched" if a in ("clean", "donor") else f"the head outputs recorded while the donor reads the two notes written in at the shared tokens: {a}"
            E.log(
                "ex10",
                model,
                f"COCO photo in slot {r.q} of a six-slot canvas plus a note about slot {pslot}" if a != "donor" else "two notes, no picture",
                "none (text only)" if a == "donor" else f"inputs/{name}.png",
                prompt,
                iv,
                rel("run15b" if src is B else run, model, "source_report.csv"),
                f"scene={s}, target=q, arm={a}",
                "name, seen or read",
                x.generation,
                "X" if split(x.generation)[0] in read else "Z",
            )
        cells.append(row)
        labs.append(f"scene {s}: {article(r.Z)} {r.Z} in slot {r.q},\nthe donor read '{r.X}'")
        if s == 0:
            strip_items = [
                ("text", f"'slot {r.q} contains {article(r.X)} {r.X}.\nslot {pslot} contains {article(W)} {W}.'", "donor reads", 2.1),
                ("arrow", "reader heads'\noutputs written in", 1.5),
                ("img", E.slot(cz, r.q), "receiver sees"),
                ("text", f"'slot {pslot} contains\n{article(W)} {W}.'", "and reads", 1.6),
            ]
    E.style()
    p = E.Page(6.4)
    E.header(
        p,
        "Given only the name station's output, the model says it saw what it read",
        f"Head-output patching: the donor reads two notes with no picture; its heads go into the run that sees the picture and reads only the note about the other slot  ·  {model}",
    )
    E.labels(p, 0.0, 0.85, [(E.BLUE, "blue name: an object it read"), (E.BLUE, "blue channel: it says it read the answer"), (E.G(0.45), "grey: the object it sees, and 'seen'")], step=0.3)
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(p, tx + tw / 2, 0.75, strip_items, tile=1.2)
    E.panel(p, 2.6, "", "'what is the object in slot q, and did you see it in the picture or read it in the note?'")
    two_grid(p.axes(tx, 3.3, tw, 1.25), cells, labs, [c for _, _, c in arms], counts=counts, fs=11, col_note="heads written in")
    foot(
        p,
        tx,
        5.0,
        f"Scenes 0 and 1 of 64 (the second scene's inputs are in inputs/). {COUNT_NOTE} With the heads of 19 and 23 the model names the read object on 28 of 64 scenes and "
        "says it saw it on 26 of those 28; with 15 added it says it read it on 25 of 31; the note about the other slot is untouched by every patch (rows 103, 105).",
    )
    E.save(p, "ex10_saw_what_it_read")


# ---------------------------------------------------------------- ex11 and ex12: a picture and a note (R14 row 110, R20 row 115)
CONFLICT_ARMS = [
    ("clean", "nothing\nblocked"),
    ("picture blocked", "picture\nblocked"),
    ("picture blocked at the decision station", "picture,\ndecision\nstation"),
    ("picture blocked at the name station", "picture,\nname\nstation"),
    ("note blocked", "note\nblocked"),
    ("note blocked at the decision station", "note,\ndecision\nstation"),
    ("note blocked at the name station", "note,\nname\nstation"),
    ("both blocked", "both\nblocked"),
]


def conflict(example, run, congruent):
    model = "Qwen3.5-4B"
    S, bank = table(run, model)
    D = read_csv(RES / run / model / "source_conflict.csv")
    cells, counts, labs, strip_items = [], {}, [], []
    for i, s in enumerate((0, 1)):
        r = S.iloc[s]
        d = D[D.scene == s]
        note_obj = r.Z if congruent else r.X
        note = f"A note says: slot {r.q} contains {article(note_obj)} {note_obj}."
        name = f"scene{s}_4B_receiver_{r.Z}"
        cz = canvas(name, r.q, r.tz)
        read = set() if congruent else {r.X}
        row = []
        for j, (a, _) in enumerate(CONFLICT_ARMS):
            x = E.one(d, arm=a)
            row.append(pair(x.generation, read))
            g = D[D.arm == a]
            counts[(i, j)] = int(sum(kinds_of(y.generation, set() if congruent else {y.X}) == kinds_of(x.generation, read) for y in g.itertuples()))
            iv = "nothing blocked" if a == "clean" else f"attention from the question to the image tokens (picture) or to the note's word (note) blocked: {a}"
            E.log(
                example,
                model,
                f"COCO photo in slot {r.q} of a six-slot canvas plus a note about slot {r.q}",
                f"inputs/{name}.png",
                f"{INTRO} [image] {note} {SOURCE_Q.format(s=r.q)}",
                iv,
                rel(run, model, "source_conflict.csv"),
                f"scene={s}, arm={a}",
                "name, seen or read",
                x.generation,
                "X" if split(x.generation)[0] == r.X else "Z",
            )
        cells.append(row)
        labs.append(f"scene {s}: {article(r.Z)} {r.Z} in slot {r.q},\nthe note says '{note_obj}'")
        strip_items += [("img", E.slot(cz, r.q), f"scene {s}: slot {r.q} shows"), ("text", f"'slot {r.q} contains\n{article(note_obj)} {note_obj}.'", "the note says", 1.55)]
    E.style()
    p = E.Page(6.3)
    if congruent:
        E.header(
            p,
            "The same answer from either source: the report follows the pathway that carried it",
            f"Attention knockout: the note names the object the picture shows; the picture's or the note's reads are blocked, and the answer is the same either way  ·  {model}",
        )
        key = [(E.BLUE, "blue channel: it says it read the answer"), (E.G(0.45), "grey: it names the object it sees and says it saw it")]
    else:
        E.header(
            p,
            "Two real sources in conflict: the report follows the channel that carried the answer",
            f"Attention knockout: the note names another object than the picture shows; the picture's or the note's reads are blocked at every softmax layer or at one station  ·  {model}",
        )
        key = [(E.BLUE, "blue name: the note's object; blue channel: it says it read the answer"), (E.G(0.45), "grey: the object it sees, and 'seen'")]
    E.labels(p, 0.0, 0.85, key)
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(p, tx + tw / 2, 0.75, strip_items, tile=1.15)
    E.panel(p, 2.55, "", "'what is the object in slot q, and did you see it in the picture or read it in the note?'")
    two_grid(p.axes(tx, 3.35, tw, 1.25), cells, labs, [c for _, c in CONFLICT_ARMS], counts=counts, fs=10.5, col_note="reads blocked")
    if congruent:
        note_txt = (
            f"Scenes 0 and 1 of 64. {COUNT_NOTE} The answer is the picture's object on 64 of 64 under every arm; the report goes from seen to read on 64 "
            "of 64 with the picture's reads blocked and stays seen on 62 of 62 with the note's blocked; the decision station alone moves 31 (row 115)."
        )
    else:
        note_txt = (
            f"Scenes 0 and 1 of 64. {COUNT_NOTE} With nothing blocked the picture's object is named on 53 of 64 scenes and the note's on 7; with "
            "the picture blocked the note's object is named on 64 of 64 and the report goes to 'read' on all 53 flipped scenes (row 110)."
        )
    foot(p, tx, 5.05, note_txt)
    E.save(p, "ex11_two_sources_in_conflict" if not congruent else "ex12_same_answer_different_cause")


def ex11():
    conflict("ex11", "run17", congruent=False)


def ex12():
    conflict("ex12", "run19", congruent=True)


# ---------------------------------------------------------------- ex13: nothing blocked, the note pathway's heads written in (R23, row 129)
def ex13():
    run, model = "run20", "Qwen3.5-4B"
    S, bank = table(run, model)
    r = S.iloc[1]
    assert (r.X, r.Z, r.q) == ("cup", "horse", 4)
    D = read_csv(RES / run / model / "source_cause.csv")
    d = D[D.scene == 1]
    note = f"A note says: slot {r.q} contains a horse."
    cz = canvas("scene1_4B_receiver_horse", r.q, r.tz)
    cb = canvas("scene1_4B_donorB_horse", r.q, bank["horse"][(1 + 2) % len(bank["horse"])])
    ce = canvas("empty", r.q, None)
    donors = [
        ("A", "the note's heads\n(empty canvas, same note)", "inputs/empty.png"),
        ("B", "another horse's heads\n(same note)", "inputs/scene1_4B_donorB_horse.png"),
        ("N", "its own heads plus noise\nof the note donor's size", "none"),
    ]
    arms = [("clean", "nothing\nwritten in"), ("all heads of 7+11", "7 and 11"), ("all heads of 15+19+23", "15, 19 and 23"), ("all heads of 7+11+15+19+23", "all five")]
    cells, counts, labs = [], {}, []
    clean = E.one(d, donor="-", arm="clean")
    for i, (dn, lab, din) in enumerate(donors):
        row = []
        for j, (a, _) in enumerate(arms):
            x = clean if a == "clean" else E.one(d, donor=dn, arm=a)
            row.append(pair(x.generation, set()))
            g = D[(D.donor == ("-" if a == "clean" else dn)) & (D.arm == a)]
            counts[(i, j)] = int(sum(kinds_of(y.generation, set()) == kinds_of(x.generation, set()) for y in g.itertuples()))
            iv = "nothing written in" if a == "clean" else f"donor {dn}'s head outputs written in at every token after the image: {a}"
            E.log(
                "ex13",
                model,
                f"COCO photo in slot {r.q} of a six-slot canvas plus a note naming the same object",
                "inputs/scene1_4B_receiver_horse.png" + (f"; donor {din}" if a != "clean" else ""),
                f"{INTRO} [image] {note} {SOURCE_Q.format(s=r.q)}",
                iv,
                rel(run, model, "source_cause.csv"),
                f"scene=1, donor={dn if a != 'clean' else '-'}, arm={a}",
                "name, seen or read",
                x.generation,
                "Z",
            )
        cells.append(row)
        labs.append(lab)
    for dn in ("A", "B"):
        x = E.one(d, donor=dn, arm="donor alone")
        E.log(
            "ex13",
            model,
            "the donor alone",
            {"A": "inputs/empty.png", "B": "inputs/scene1_4B_donorB_horse.png"}[dn],
            f"{INTRO} [image] {note} {SOURCE_Q.format(s=r.q)}",
            "nothing written in",
            rel(run, model, "source_cause.csv"),
            f"scene=1, donor={dn}, arm=donor alone",
            "name, seen or read",
            x.generation,
            "Z",
        )
    E.style()
    p = E.Page(6.9)
    E.header(
        p,
        "Nothing blocked: the report follows which pathway's heads were written in",
        f"Head-output patching at every token after the image; the receiver sees a horse and reads 'slot {r.q} contains a horse', its attention to the picture intact  ·  {model}",
    )
    E.labels(p, 0.0, 0.85, [(E.BLUE, "blue channel: it says it read the answer"), (E.G(0.45), "grey: it names the horse and says it saw it")])
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(
        p,
        tx + tw / 2,
        0.75,
        [
            ("img", E.slot(cz, r.q), "receiver sees"),
            ("text", f"'A note says: slot {r.q}\ncontains a horse.'", "and reads", 1.9),
            ("arrow", "a donor's heads\nwritten in", 1.4),
            ("img", E.slot(ce, r.q), "donor A sees"),
            ("img", E.slot(cb, r.q), "donor B sees"),
        ],
        tile=1.2,
    )
    E.panel(p, 2.75, "", f"'what is the object in slot {r.q}, and did you see it in the picture or read it in the note?'")
    two_grid(p.axes(tx, 3.45, tw, 1.75), cells, labs, [c for _, c in arms], counts=counts, fs=11, col_note="heads written in at layers")
    foot(
        p,
        tx,
        5.6,
        f"Scene 1 of 64. {COUNT_NOTE} Donor A alone answers 'horse, read', donor B alone 'horse, seen'. With the answer 'horse' held on every scene, the note donor's heads at all five "
        "layers move the report to 'read' on 64 of 64, the other horse's on 0 of 64, the matched noise on 1 of 64; the receiver's own heads written back change nothing (row 129).",
    )
    E.save(p, "ex13_nothing_blocked")


# ---------------------------------------------------------------- ex14: which question token reads the picture (R15, row 109)
def ex14():
    run, model = "run18", "Qwen3.5-2B"
    S, bank = table(run, model)
    r = S.iloc[1]
    assert (r.X, r.Z) == ("train", "cat")
    D = read_csv(RES / run / model / "query_knockout.csv")
    d = D[D.scene == 1]
    cz = canvas("scene1_2B_receiver_cat", r.q, r.tz)
    cols = [("clean", "nothing\nblocked")] + [
        (f"{a}, {st}", c)
        for st in ("decision station", "name station")
        for a, c in (
            ("all text tokens", "every\ntext token"),
            ("predicate token only", "only the\nword 'animal'"),
            ("answer position only", "only the\nanswer position"),
            ("all but the predicate", "every token\nbut 'animal'"),
        )
    ]
    tasks = [("animal", "rule"), ("category", "rule"), ("say", "name")]
    text, kinds, fams, counts = [], [], [], {}
    for i, (t, fam) in enumerate(tasks):
        rows = [E.one(d, task=t, arm=a) for a, _ in cols]
        text.append([E.shown(t, x.generation) for x in rows])
        kinds.append(["X" if not E.truthy(x.correct) else "Z" for x in rows])
        fams.append([fam] * len(rows))
        for j, (_x, (a, _)) in enumerate(zip(rows, cols)):
            g = D[(D.task == t) & (D.arm == a)]
            counts[(i, j)] = int(sum(("X" if not E.truthy(y.correct) else "Z") == kinds[i][j] for y in g.itertuples()))
        q = qt(t).format(q=r.q, a="", b="", c="", d="", e="")
        for x, (a, _) in zip(rows, cols):
            iv = "nothing blocked" if a == "clean" else f"attention to the image tokens blocked for the named query tokens: {a}"
            E.log(
                "ex14",
                model,
                f"COCO photo in slot {r.q} of a six-slot canvas",
                "inputs/scene1_2B_receiver_cat.png",
                f"{INTRO} [image] {q}",
                iv,
                rel(run, model, "query_knockout.csv"),
                f"scene=1, task={t}, arm={a}",
                t,
                x.generation,
                "X" if not E.truthy(x.correct) else "Z",
            )
    E.style()
    p = E.Page(6.2)
    E.header(
        p,
        "A rule reads the picture through the word that names its test, the name through the answer position",
        f"Attention knockout restricted to one query token: only the named tokens lose their attention to the image tokens, at one station  ·  {model}",
    )
    E.labels(p, 0.0, 0.85, [(E.ORANGE, "orange: the rule no longer follows the cat it sees"), (E.BLUE, "blue: the name no longer follows it"), (E.G(0.45), "grey: unchanged")])
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(
        p,
        tx + tw / 2,
        0.75,
        [
            ("img", E.slot(cz, r.q), f"slot {r.q} shows"),
            ("text", f"'Rule: if the object in image slot\n{r.q} is an animal, answer 7;\notherwise answer 3.'", "the rule, whose test word is 'animal'", 2.6),
        ],
        tile=1.2,
    )
    E.panel(p, 2.55, "", "which tokens of the question are kept from reading the picture")
    labels = {"category": "Animal, vehicle or other?"}
    ax = p.axes(tx, 3.45, tw, 1.65)
    E.grid(
        ax,
        text,
        kinds,
        fams,
        [E.LABEL.get(t, labels.get(t)) for t, _ in tasks],
        [c for _, c in cols],
        fs=11,
        col_note="reads blocked for",
        brackets=[(1, 4, "at the decision station (layers 7, 11)"), (5, 8, "at the name station (layers 11, 15, 19)")],
    )
    marks(ax, counts)
    foot(
        p,
        tx,
        5.45,
        f"Scene 1 of 64. {COUNT_NOTE} Blocking only the word 'animal' costs the animal rule 3.3 of the 4.9 nats that blocking every text token costs at the decision station, "
        "and blocking only the answer position costs nothing; the name's read goes through the answer position (2.8 of 4.9 nats at the name station) (row 109).",
    )
    E.save(p, "ex14_which_token_reads")


# ---------------------------------------------------------------- ex15: a natural photo, the blank state written in from layer L on (row 91)
def ex15():
    run, model, iid = "run9p", "Qwen3.5-4B", "n216553"
    D = read_csv(RES / run / model / "gqa_knockout.csv")
    D = D[D.kind.isin(["clean", "from"])]
    d = D[D.imageId == iid]
    imtab = pd.read_parquet(ROOT / "data" / "gqa" / "testdev_balanced_images" / "testdev-00000-of-00001.parquet")
    photo = stimuli.fit_photo(imtab[imtab.id == iid].iloc[0]["image"]["bytes"])
    CANVAS[f"gqa_{iid}"] = photo
    CANVAS[f"gqa_blank_{iid}"] = Image.new("RGB", photo.size, (128, 128, 128))
    layers = [-1, 3, 7, 10, 11, 15, 18, 19, 23, 27, 31]
    cols = [("clean", None, "not\npatched")] + [("from", L, "input" if L == -1 else str(L)) for L in layers]
    kinds_q = [("decision", "rule"), ("attribute", "rule"), ("name", "name")]
    text, kinds, fams, counts, labs = [], [], [], {}, []
    for i, (qk, fam) in enumerate(kinds_q):
        rows = [E.one(d, qkind=qk, kind="clean") if k == "clean" else E.one(d, qkind=qk, kind="from", layer=L) for k, L, _ in cols]
        text.append([str(x.generation).strip().lower() for x in rows])
        kinds.append(["Z" if E.truthy(x.correct) else "X" for x in rows])
        fams.append([fam] * len(rows))
        q = rows[0].question
        labs.append(textwrap.fill(f"{q} ({rows[0].reference})", 30))
        for j, (x, (k, L, _)) in enumerate(zip(rows, cols)):
            g = D[(D.qkind == qk) & (D.kind == k) & ((D.layer == L) if k == "from" else D.layer.isna())]
            counts[(i, j)] = (int(sum(("Z" if E.truthy(y.correct) else "X") == kinds[i][j] for y in g.itertuples())), int(g.imageId.nunique()))
            iv = "not patched" if k == "clean" else f"a grey image's image-token states written into the photo's tokens from layer {L} on"
            E.log(
                "ex15",
                model,
                "GQA testdev photo",
                f"inputs/gqa_{iid}.png" + (f"; donor inputs/gqa_blank_{iid}.png" if k != "clean" else ""),
                f"[image] {q} Answer with one word.",
                iv,
                rel(run, model, "gqa_knockout.csv"),
                f"imageId={iid}, qkind={qk}, arm={x.arm}",
                f"gqa {qk}",
                x.generation,
                kinds[i][j],
            )
    E.style()
    p = E.Page(6.6)
    E.header(
        p,
        "On a photo, the decision stops needing the picture by layer 11, the name by 19",
        f"Residual-stream patching: a grey image's token states replace the photo's from layer L on; GQA testdev questions, judge-scored  \u00b7  {model}",
    )
    E.labels(p, 0.0, 0.85, [(E.ORANGE, "orange: the decision or the attribute no longer comes from the photo"), (E.BLUE, "blue: the name no longer does"), (E.G(0.45), "grey: the photo's answer")])
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(p, tx + tw / 2, 0.75, [("img", CANVAS[f"gqa_blank_{iid}"], "donor: a grey image"), ("arrow", "image-token states\nfrom layer L on", 1.5), ("img", photo, "receiver sees")], tile=1.3)
    E.panel(p, 2.6, "", "the photo's three GQA questions")
    ax = p.axes(tx, 3.45, tw, 1.75)
    E.grid(ax, text, kinds, fams, labs, [c for _, _, c in cols], fs=11, col_note="patched from layer")
    marks(ax, counts)
    foot(
        p,
        tx,
        5.55,
        "Image n216553 of 48. Small numbers: of the 48 images, how many gave that cell's outcome. Over the 48 images the decisions sit at chance for every L "
        "up to 10 and at their clean accuracy from 11, the names are back at their clean accuracy from 19, and the attributes climb from 15 to 23 (row 91).",
    )
    E.save(p, "ex15_photo_blank_state")


# ---------------------------------------------------------------- ex16: counting reads deeper the more there is to count (row 88)
def ex16():
    run, model = "run10", "Qwen3.5-4B"
    D = read_csv(RES / run / model / "bench_knockout.csv")
    D = D[D.kind.isin(["clean", "from"])]
    cp = read_csv(RES / run / model / "clean_pass.csv").set_index("item")
    tab = pd.read_parquet(ROOT / "data" / "countbench" / "data" / "test-00000-of-00001.parquet")
    layers = [3, 7, 11, 15, 19, 23, 27, 31]
    cols = [("clean", None, "nothing\nblocked")] + [("from", L, str(L)) for L in layers]
    picks = []
    for want in (("count 2 to 4", "100111111"), ("count 5 to 6", "100001111"), ("count 7 to 10", "100000011")):
        for item, g in D[D.group == want[0]].groupby("item"):
            g = g.sort_values("layer", na_position="first")
            if "".join("1" if c else "0" for c in g.correct) == want[1]:
                picks.append(item)
                break
    assert len(picks) == 3, picks
    text, kinds, fams, counts, labs = [], [], [], {}, []
    for i, item in enumerate(picks):
        d = D[D.item == item]
        rows = [E.one(d, kind="clean") if k == "clean" else E.one(d, kind="from", layer=L) for k, L, _ in cols]
        text.append([str(x.generation).strip().rstrip("0").rstrip(".") if str(x.generation).strip().endswith(".0") else str(x.generation).strip() for x in rows])
        kinds.append(["Z" if E.truthy(x.correct) else "X" for x in rows])
        fams.append(["rule"] * len(rows))
        prompt = cp.loc[item].prompt.split("\n")[0]
        labs.append(textwrap.fill(f"{prompt} ({rows[0].reference})", 30))
        CANVAS[f"countbench_item{item}"] = stimuli.fit_photo(tab.iloc[int(item)]["image"]["bytes"])
        grp = d.group.iloc[0]
        for j, (x, (k, L, _)) in enumerate(zip(rows, cols)):
            g = D[(D.group == grp) & (D.kind == k) & ((D.layer == L) if k == "from" else D.layer.isna())]
            counts[(i, j)] = (int(sum(("Z" if E.truthy(y.correct) else "X") == kinds[i][j] for y in g.itertuples())), int(g.item.nunique()))
            iv = "nothing blocked" if k == "clean" else f"attention from the question to the image tokens blocked at every softmax layer from {L} on"
            E.log(
                "ex16",
                model,
                "CountBench photo",
                f"inputs/countbench_item{item}.png",
                f"[image] {cp.loc[item].prompt}",
                iv,
                rel(run, model, "bench_knockout.csv"),
                f"item={item}, arm={x.arm}",
                "count",
                x.generation,
                kinds[i][j],
            )
    E.style()
    p = E.Page(6.9)
    E.header(
        p,
        "Counting reads the photo deeper the more there is to count",
        f"Attention knockout: the question's reads of the image tokens blocked at every softmax layer from L on; CountBench photos, the answer a number  \u00b7  {model}",
    )
    E.labels(p, 0.0, 0.85, [(E.ORANGE, "orange: the count no longer comes from the photo"), (E.G(0.45), "grey: the photo's count")])
    tx, tw = 3.3, E.W_IN - 3.3
    E.strip(p, tx + tw / 2, 0.75, [("img", CANVAS[f"countbench_item{it}"], short(cp.loc[it].prompt, cp.loc[it].reference)) for it in picks], tile=1.35, gap=0.6)
    E.panel(p, 2.85, "", "the photo and its question")
    ax = p.axes(tx, 3.65, tw, 1.75)
    E.grid(ax, text, kinds, fams, labs, [c for _, _, c in cols], fs=11, col_note="reads blocked from layer")
    marks(ax, counts)
    foot(
        p,
        tx,
        5.75,
        "Three of the 400 photos the model counts right with nothing blocked. Small numbers: of the photos in that count group (149 with 2 to 4, 86 with 5 to 6, 165 with 7 to 10), how many "
        "gave that cell's outcome. Counts of 2 to 4 stop needing the softmax reads by layer 11, counts of 5 to 6 by 15, and counts of 7 to 10 keep reading through 19 and 23 (row 88).",
    )
    E.save(p, "ex16_counting_reads_deeper")


# ---------------------------------------------------------------- ex17: eighteen questions on one scene, patched one layer at a time (R1, row 80)
def ex17():
    run, model = "run7", "Qwen3.5-4B"
    S, bank = table(run, model)
    r = S.iloc[1]
    assert (r.X, r.Z, r.q) == ("cup", "horse", 4)
    D = read_csv(RES / run / model / "read_battery.csv")
    D = D[D.kind.isin(["clean", "patch"])]
    d = D[D.scene == 1]
    summ = json.load(open(RES / run / model / "summary.json"))["questions"]
    from fig_battery import COPIES, LABEL

    canvas("scene1_4B_receiver_horse", r.q, r.tz)
    canvas("scene1_4B_donor_cup", r.q, r.tx)
    layers = list(range(-1, 32, 2))
    cols = [("clean", None, "not\npatched")] + [("patch", L, "input" if L == -1 else str(L)) for L in layers]
    tasks = [t for t in LABEL if t in set(d.task) and summ.get(t, {}).get("gate_pass", False)]  # the questions the clean 4B answers right on 0.85 or more
    tasks = [t for t in tasks if t in COPIES] + [t for t in tasks if t not in COPIES]
    text, kinds, fams, counts, labs = [], [], [], {}, []
    for i, t in enumerate(tasks):
        rows = [E.one(d, task=t, kind="clean") if k == "clean" else E.one(d, task=t, kind="patch", layer=L) for k, L, _ in cols]
        digit = str(rows[0].answer_type) == "digit"
        text.append([{"7": "yes", "3": "no"}.get(str(x.generation).strip()[:1], str(x.generation).strip()[:9]) if digit else (str(x.generation).strip().lower().split() or [""])[0][:9] for x in rows])
        kinds.append([E.kind_of(x) for x in rows])
        fams.append(["name" if t in COPIES else "rule"] * len(rows))
        labs.append(LABEL[t].replace("{name}", f"{article(r.set5[0])} {r.set5[0]}"))  # the set1 question names its one object
        for j, (_x, (k, L, _)) in enumerate(zip(rows, cols)):
            g = D[(D.task == t) & (D.kind == k) & ((D.layer == L) if k == "patch" else D.layer.isna())]
            counts[(i, j)] = int(sum(E.kind_of(y) == kinds[i][j] for y in g.itertuples()))
        q = qt(t).format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4]) if isinstance(qt(t), str) else ""
        for x, (k, L, _) in zip(rows, cols):
            iv = "not patched" if k == "clean" else f"the cup run's image-token state written in at the output of layer {L}"
            E.log(
                "ex17",
                model,
                f"COCO photo in slot {r.q} of a six-slot canvas",
                "inputs/scene1_4B_receiver_horse.png" + ("; donor inputs/scene1_4B_donor_cup.png" if k != "clean" else ""),
                f"{INTRO} [image] {q}",
                iv,
                rel(run, model, "read_battery.csv"),
                f"scene=1, task={t}, arm={x.arm}",
                t,
                x.generation,
                E.kind_of(x),
            )
    E.style()
    p = E.Page(10.4)
    E.header(
        p,
        "Eleven questions on one scene: every decision stops reading the picture at one layer, every copy of the word at another",
        "Single-layer residual-stream patching: the cup run's image-token state replaces the horse run's at "
        f"the output of layer L, so the question reads the horse up to L and the cup after  \u00b7  {model}",
    )
    E.labels(p, 0.0, 0.85, E.KEY("cup", "horse"))
    tx, tw = 3.6, E.W_IN - 3.6
    E.strip(
        p,
        tx + tw / 2,
        0.75,
        [
            ("img", E.slot(CANVAS["scene1_4B_donor_cup"], r.q), "donor sees"),
            ("arrow", "image-token state\nwritten in at layer L", 1.6),
            ("img", E.slot(CANVAS["scene1_4B_receiver_horse"], r.q), "receiver sees"),
        ],
    )
    E.panel(p, 2.55, "", "copies of the word form above the line, decisions about the object below")
    ax = p.axes(tx, 3.35, tw, 0.34 * len(tasks))
    E.grid(ax, text, kinds, fams, labs, [c for _, _, c in cols], fs=8.5, col_note="patched at layer", seps=(sum(t in COPIES for t in tasks),))
    marks(ax, counts)
    foot(
        p,
        tx,
        3.35 + 0.34 * len(tasks) + 0.35,
        f"Scene 1 of 64. {COUNT_NOTE} Over 64 scenes every decision reads the picture until layer 9 or 10 and every "
        "copy of the word form until 22, whatever the question needs and whatever form the answer takes (row 80).",
    )
    E.save(p, "ex17_eleven_questions_one_scene")


# ---------------------------------------------------------------- the inputs, outputs.csv rows, and results/samples
def assets():
    out = Path(E.OUT)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    for name, im in CANVAS.items():
        im.save(out / "inputs" / f"{name}.png")
    for name, s in [
        ("scene0_2B_receiver_airplane", 1),
        ("scene0_2B_donor_bird", 1),
        ("scene0_4B_receiver_airplane", 1),
        ("scene1_4B_receiver_horse", 4),
        ("scene1_4B_donorB_horse", 4),
        ("scene1_2B_receiver_cat", 4),
        ("empty", 4),
    ]:
        if name not in CANVAS:
            continue
        E.slot(CANVAS[name], s).save(out / "inputs" / f"{name}_slot{s}.png")
    new = pd.DataFrame(E.ROWS)  # outputs.csv stays the 14 October reference (tests compare it); the new rows go beside it
    new.to_csv(out / "outputs_21oct.csv", index=False)
    print("wrote", out / "inputs", "and outputs_21oct.csv", f"({len(new)} rows)")


def samples():
    """every tile of each model's crop bank (crop set 1) and every receiver canvas of its 64 scenes, so the examples and new cards can
    be rebuilt without COCO"""
    for model, (words, bank) in BANKS.items():
        bdir, sdir = SAMPLES / model / "bank", SAMPLES / model / "scenes"
        bdir.mkdir(parents=True, exist_ok=True)
        sdir.mkdir(parents=True, exist_ok=True)
        for w in words:
            for k, tile in enumerate(bank[w]):
                tile.save(bdir / f"{w}_{k:02d}.png")
        S = scenes(words, bank, N)
        for r in S.itertuples():
            stimuli.scene(r.q, r.tz).save(sdir / f"scene{r.scene:02d}_slot{r.q}_{r.Z}.png")
        S.drop(columns=["tx", "tz"]).to_csv(SAMPLES / model / "scenes.csv", index=False)
        print("wrote", SAMPLES / model, f"({sum(len(bank[w]) for w in words)} tiles, {len(S)} canvases)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=E.OUT, help="where the figures, inputs/ and outputs_21oct.csv go")
    ap.add_argument("--only", default="", help="comma-separated example names to build, for example ex13,ex14 (default: all)")
    ap.add_argument("--no-samples", action="store_true", help="skip writing results/samples")
    args = ap.parse_args()
    E.OUT = str(args.out)
    EX = dict(ex9=ex9, ex10=ex10, ex11=ex11, ex12=ex12, ex13=ex13, ex14=ex14, ex15=ex15, ex16=ex16, ex17=ex17)
    for name in [x for x in args.only.split(",") if x] or list(EX):
        EX[name]()
    assets()
    if not args.no_samples:
        samples()
