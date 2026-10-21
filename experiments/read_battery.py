"""Read battery: at which layer does each question stop reading the object's image tokens, and what sets that depth? (R1 in RECORD.md)

The depth sweep of D1 at single-layer resolution, with log-probabilities, over a battery of questions that cross what the question
needs from the object (its category, its word form, world knowledge, a logical rule) with what the answer is (a digit, a letter,
a word). A donor run sees object X in slot q of the six-slot canvas (no question); a receiver sees Z of the other kind and is asked
one question. The donor's image-token residual states replace the receiver's at the output of one layer L (L = -1 is the input):
every later layer then rebuilds the donor's image states, so the text positions read X from layer L + 1 on and Z up to L. The
answer follows X only while the question still reads the image tokens after layer L, so the last such L is the question's read
depth. Per question and layer the script saves the generated answer and the log-probability of X's answer and of Z's answer at
the answer position (first token, summed over its single-token forms), so each curve is both a flip rate and a shift in nats.

Dose-response (--mix): the donor's state mixed into the receiver's at weight a (0.25, 0.5, 0.75) at one layer, and at 0.5 with
the mixed vector rescaled to the receiver's norm per token. If reading were a smooth sum over many layers, the cutoff would move
earlier as the donor weakens; if it is a few readers at fixed depths, the cutoff stays and the plateau falls.

Scoring: word answers by the first vocabulary word (synonym list in ANSWERS), letters and digits by the first character or number.
A scene counts for a question only when X's and Z's answers differ. A question is kept for a model only when the receiver alone
answers it correctly on >= 0.85 of scenes (frozen gate); dropped questions are listed in summary.json.

Outputs in runs/<run>/<model>/: read_battery.csv (task x arm x scene: generation, follows_X, keeps_Z, lp_X, lp_Z, ld), summary.json
(per question: gate, n, curves by layer with scene-bootstrap intervals, read depth by flips and by log-probability, transition
width; dose-response), crops.json, stimuli.png.
"""

import json
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import torch

from jspace import hooks, stimuli
from jspace.analysis import boot_scene
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
VEHICLES = {"car", "bus", "train", "bicycle", "airplane", "boat"}
INTRO = "Study the scene. It has an image grid with numbered slots."
PLURAL = {"bus": "buses"}
FRENCH = {
    "dog": "chien",
    "cat": "chat",
    "car": "voiture",
    "bus": "bus",
    "chair": "chaise",
    "clock": "horloge",
    "cup": "tasse",
    "bird": "oiseau",
    "train": "train",
    "horse": "cheval",
    "bicycle": "velo",
    "airplane": "avion",
    "boat": "bateau",
    "bed": "lit",
    "pizza": "pizza",
    "laptop": "ordinateur",
}
LEGS = {"dog": 4, "cat": 4, "bird": 2, "horse": 4, "chair": 4, "bed": 4}
SYN = {
    "cup": ["mug"],
    "bird": ["seagull", "pigeon", "parrot", "duck"],
    "airplane": ["plane", "jet", "aircraft"],
    "laptop": ["computer", "notebook"],
    "bicycle": ["bike"],
    "dog": ["puppy"],
    "cat": ["kitten"],
    "horse": ["pony"],
}

# question -> (template, what it needs, answer type); {q} the slot, {a}..{e} the names of the set rules
QUESTIONS = {
    "say": ("What is the object in image slot {q}? Reply with the noun only.", "word form", "word"),
    "plural": ("What is the object in image slot {q}? Reply with the plural noun only.", "word form", "word"),
    "french": ("What is the object in image slot {q}? Reply with the French noun only, without an article.", "word form", "word"),
    "letter": ("What is the first letter of the name of the object in image slot {q}? Reply with the letter only.", "word form", "letter"),
    "last": ("What is the last letter of the name of the object in image slot {q}? Reply with the letter only.", "word form", "letter"),
    "nletters": ("How many letters are in the name of the object in image slot {q}? Reply with the number only.", "word form", "digit"),
    "long": ("Rule: if the name of the object in image slot {q} has more than four letters, answer 7; otherwise answer 3. Reply with the number only.", "word form", "digit"),
    "before_d": (
        "Rule: if the first letter of the name of the object in image slot {q} comes before D in the alphabet, answer 7; otherwise answer 3. Reply with the number only.",
        "word form",
        "digit",
    ),
    "set1": ("Rule: if the object in image slot {q} is a {a}, answer 7; otherwise answer 3. Reply with the number only.", "word form", "digit"),
    "set": ("Rule: if the object in image slot {q} is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.", "word form", "digit"),
    "set5": ("Rule: if the object in image slot {q} is a {a}, a {b}, a {c}, a {d} or a {e}, answer 7; otherwise answer 3. Reply with the number only.", "word form", "digit"),
    "animal": ("Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.", "category", "digit"),
    "notanimal": ("Rule: if the object in image slot {q} is not an animal, answer 7; otherwise answer 3. Reply with the number only.", "category", "digit"),
    "ride": ("Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.", "world knowledge", "digit"),
    "legs": ("How many legs does the object in image slot {q} have? Reply with the number only.", "world knowledge", "digit"),
    "andrule": (
        "Rule: if the object in image slot {q} is an animal and its name has more than four letters, answer 7; otherwise answer 3. Reply with the number only.",
        "category and word form",
        "digit",
    ),
    "category": ("Is the object in image slot {q} an animal, a vehicle, or something else? Reply with one word: animal, vehicle or other.", "category", "word"),
    "yesno": ("Is the object in image slot {q} an animal? Reply with yes or no.", "category", "word"),
    # a 16-way identity decision whose answer is a letter from a fixed list, never the word form (R11; the resolution control)
    "mc": (
        "Which of these is the object in image slot {q}? " + ", ".join(f"{chr(65 + i)}: {w}" for i, w in enumerate(sorted(stimuli.VOCAB))) + ". Reply with the letter only.",
        "identity, 16 ways",
        "letter",
    ),
}
TWO_BY_TWO = ["say", "letter", "last", "nletters", "long", "animal", "ride", "set", "category", "yesno"]  # the cheapest decisive block


def with_distractors(r, tile, k, bank, words):
    """the scene with the object's tile in slot q and k other objects' crops in k other slots, the same for the donor and the receiver
    of a scene (seeded by the scene index); k = 0 is the plain one-object canvas"""
    if k == 0:
        return stimuli.scene(r.q, tile)
    rng = np.random.default_rng(1000 + int(r.scene))
    others = [w for w in words if w not in (r.X, r.Z)]
    picks = [others[j] for j in rng.choice(len(others), k, replace=False)]
    free = [s for s in range(1, stimuli.N_SLOTS + 1) if s != r.q]
    slots = [free[j] for j in rng.choice(len(free), k, replace=False)]
    tiles = {s: bank[w][int(r.scene) % len(bank[w])] for s, w in zip(slots, picks)}
    tiles[r.q] = tile
    return stimuli.canvas(stimuli.COLS, stimuli.ROWS, {s: tiles.get(s) for s in range(1, stimuli.N_SLOTS + 1)})


def plural(w):
    return PLURAL.get(w, w + "s")


def truth(task, r, w):
    """the correct answer to `task` about object w in scene r (a string: a word, a letter or a number)"""
    if task == "say":
        return w
    if task == "plural":
        return plural(w)
    if task == "french":
        return FRENCH[w]
    if task == "letter":
        return w[0].upper()
    if task == "last":
        return w[-1].upper()
    if task == "nletters":
        return str(len(w))
    if task == "long":
        return "7" if len(w) > 4 else "3"
    if task == "before_d":
        return "7" if w[0] < "d" else "3"
    if task == "set1":
        return "7" if w == r.set5[0] else "3"
    if task == "set":
        return "7" if w in r.set5[:3] else "3"
    if task == "set5":
        return "7" if w in r.set5 else "3"
    if task == "animal":
        return "7" if w in ANIMALS else "3"
    if task == "notanimal":
        return "3" if w in ANIMALS else "7"
    if task == "ride":
        return "7" if w in RIDDEN else "3"
    if task == "legs":
        return str(LEGS.get(w, 0))
    if task == "andrule":
        return "7" if (w in ANIMALS and len(w) > 4) else "3"
    if task == "category":
        return "animal" if w in ANIMALS else "vehicle" if w in VEHICLES else "other"
    if task == "yesno":
        return "yes" if w in ANIMALS else "no"
    if task == "mc":
        return chr(65 + sorted(stimuli.VOCAB).index(w))
    raise ValueError(task)


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def matches(task, answer, target, words):
    """does the generated answer give `target`? word answers by their first vocabulary word (synonyms count for say), letters by the
    first letter, numbers by the first integer"""
    a = strip_accents(answer).lower().strip().strip("*").strip()
    kind = QUESTIONS[task][2]
    if kind == "letter":
        m = re.search(r"[a-z]", a)
        return bool(m) and m.group(0).upper() == target
    if kind == "digit":
        m = re.search(r"\d+", a)
        return bool(m) and m.group(0) == target
    toks = re.findall(r"[a-z]+", a)
    if task == "say":
        vocab = set(words) | {s for w in words for s in SYN.get(w, [])}
        first = next((t for t in toks if t in vocab or t.rstrip("s") in vocab), None)
        if first is None:
            return False
        first = first if first in vocab else first.rstrip("s")
        return first == target or first in SYN.get(target, [])
    if task == "plural":
        vocab = {plural(w) for w in words} | set(words)
        first = next((t for t in toks if t in vocab), None)
        return first == target
    if task == "french":
        vocab = set(FRENCH.values())
        first = next((t for t in toks if t in vocab), None)
        return first == target
    if task == "category":
        first = next((t for t in toks if t in {"animal", "vehicle", "other", "something", "else"}), None)
        return (first == target) or (target == "other" and first in {"something", "else"})
    if task == "yesno":
        first = next((t for t in toks if t in {"yes", "no"}), None)
        return first == target
    raise ValueError(task)


def first_token_ids(lm, answer, kind):
    """single-token forms of the first token of an answer: the word, with a leading space, capitalized, both"""
    if kind == "letter":
        forms = [answer, " " + answer, answer.lower(), " " + answer.lower()]
    elif kind == "digit":
        forms = [answer, " " + answer]
    else:
        forms = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    return sorted({lm.tokenizer.encode(f, add_special_tokens=False)[0] for f in forms})


def patch_mix(layers, rec, pos, alpha, norm_match=False):
    """the donor's state mixed into the receiver's: x <- (1 - alpha) x + alpha D at the given positions and layers, optionally
    rescaled per token to the receiver's norm (a copy of hooks.patch with the mix)"""

    def build(enc, idx):
        dev = enc["input_ids"].device
        b, s = enc["input_ids"].shape
        idx_ = np.asarray(idx)
        Pm = torch.as_tensor(np.stack([pos[i](s) for i in idx_]), device=dev)
        rows = torch.arange(b, device=dev)[:, None].expand(-1, Pm.shape[1])
        out = []
        for L in layers:
            D = torch.as_tensor(np.asarray(rec[L])[idx_], device=dev, dtype=torch.float32)

            def edit(x, D=D):
                v = x[rows, Pm]
                assert v.shape == D.shape, (v.shape, D.shape)
                m = (1 - alpha) * v + alpha * D
                if norm_match:
                    m = m * (v.norm(dim=-1, keepdim=True) / m.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                x[rows, Pm] = m

            out.append((L, hooks._residual_hook(edit)))
        return out

    return build


@torch.no_grad()
def generate_scored(lm, texts, images, n_new, hooks, ids_x, ids_z, vocab_ids=None):
    """greedy continuations and, from the same pass, the log-probability of X's and Z's first answer token (summed over forms), and,
    when vocab_ids is given ({word: ids}), the log-probability of every vocabulary name's first token (the answer distribution)"""

    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        handles = [lm.register(L, f) for L, f in (hooks(enc, idx) if hooks else [])]
        try:
            g = lm.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
        res = []
        for j, i in enumerate(idx):
            text = lm.tokenizer.decode(g.sequences[j, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            dist = {w: round(float(torch.logsumexp(lp[j, v], 0)), 3) for w, v in vocab_ids.items()} if vocab_ids else None
            res.append((text, float(torch.logsumexp(lp[j, ids_x[i]], 0)), float(torch.logsumexp(lp[j, ids_z[i]], 0)), dist))
        return res

    return lm.run_batches(texts, images, fn)


def main():
    ap = parser(__doc__, "read_battery")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--distractors", type=int, default=0, help="other objects' crops in other slots (R12, the load test)")
    ap.add_argument("--questions", default="all", help="'all', '2x2', or a comma list of question names")
    ap.add_argument("--step", type=int, default=1, help="layer step of the sweep (1 = every layer)")
    ap.add_argument("--mix", action="store_true", help="dose-response arms for say and animal")
    ap.add_argument("--steer", default="", help="two layer ranges a:b,c:d: steer Z's name direction on every image token at 4 sigma in each range (the timing test of G1)")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "read_battery")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank = crops.words, crops.tiles
    tasks = list(QUESTIONS) if args.questions == "all" else TWO_BY_TWO if args.questions == "2x2" else args.questions.split(",")
    t0 = time.time()

    # the scenes of D1 (the same draws, so the five original questions land on the same donor / receiver pairs), plus five set names
    rng = np.random.default_rng(17)
    rows = []
    for i in range(args.n):
        pool = [w for w in words if (w in ANIMALS) == (i % 2 == 0)]
        x = pool[int(rng.integers(len(pool)))]
        zs = [w for w in words if (w in ANIMALS) != (x in ANIMALS) and (w in RIDDEN) != (x in RIDDEN) and w[0] != x[0]]
        z = zs[int(rng.integers(len(zs)))]
        q = int(rng.integers(1, stimuli.N_SLOTS + 1))
        rng.choice([s for s in range(1, stimuli.N_SLOTS + 1) if s != q])
        others = [w for w in words if w not in (x, z)]
        fill = [others[j] for j in rng.choice(len(others), 2, replace=False)]
        sset = [x if i % 4 < 2 else z] + fill
        rng.shuffle(sset)
        rest = [w for w in others if w not in fill]
        set5 = list(sset) + [rest[j] for j in rng.choice(len(rest), 2, replace=False)]
        rows.append(dict(scene=i, X=x, Z=z, q=q, set5=set5, tx=bank[x][i % len(bank[x])], tz=bank[z][(i + 1) % len(bank[z])]))
    S = pd.DataFrame(rows)
    don_img = [with_distractors(r, r.tx, args.distractors, bank, words) for r in S.itertuples()]
    rec_img = [with_distractors(r, r.tz, args.distractors, bank, words) for r in S.itertuples()]
    stimuli.contact_sheet(
        [(f"scene {r.scene} donor {r.X} slot {r.q}", don_img[j]) for j, r in enumerate(S.itertuples()) if j < 2]
        + [(f"scene {r.scene} receiver {r.Z} slot {r.q}", rec_img[j]) for j, r in enumerate(S.itertuples()) if j < 2],
        2,
        run.path("stimuli.png"),
    )
    don_txt = [lm.chat_media([INTRO, "IMAGE"])] * len(S)
    ids0 = lm.encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
    img_pos = [j for j, t in enumerate(ids0) if t == lm.image_token_id]
    n_img = len(img_pos)
    LF = lm.all_layers()
    final, d_model = lm.final, lm.d_model
    REC = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, Rr = lm.forward(don_txt[j : j + 1], don_img[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = Rr[L][0].half().numpy()
    print(f"donor states captured ({len(LF)} layers, {n_img} image tokens)  [{time.time() - t0:.0f}s]")
    sweep = [L for L in range(-1, final + 1, args.step)]
    if final not in sweep:
        sweep.append(final)

    res = []
    for task in tasks:
        tmpl, needs, kind = QUESTIONS[task]
        TX = [truth(task, r, r.X) for r in S.itertuples()]
        TZ = [truth(task, r, r.Z) for r in S.itertuples()]
        ids_x = [first_token_ids(lm, t, kind) for t in TX]
        ids_z = [first_token_ids(lm, t, kind) for t in TZ]
        rec_txt = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4])]) for r in S.itertuples()]
        n_all = [len(lm.encode(rec_txt[j : j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
        pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
        arms = [("clean", None, "clean", None, 0.0)]
        arms += [(f"donor state at layer {L}", hooks.patch([L], REC, pos), "patch", L, 1.0) for L in sweep]
        if args.steer and task in ("say", "animal", "category", "set"):
            from jspace import lens as lens_

            at_image = hooks.image_positions(lm.image_token_id)
            ids_x = [crops.ids[w][0] for w in S.X]
            nearest = lambda L: L if (0 <= L < final and L in lm.jac) else (0 if L < 0 else final - 1)  # noqa: E731
            UX = {L: lens_.word_directions(lm, nearest(L), ids_x, "J") for L in LF}
            UR = {L: lens_.unit_rows(len(S), d_model, 900 + L) for L in LF}
            sigma = {L: float(np.mean([np.abs(REC[L][i].astype(np.float32) @ UX[L][i, :, 0]).mean() for i in range(len(S))])) for L in LF}
            for rng_ in args.steer.split(","):
                a_, b_ = (int(v) for v in rng_.split(":"))
                layers_ = [L for L in LF if a_ <= L <= b_]
                arms.append((f"steer X's name, layers {a_} to {b_}", hooks.add_directions(UX, at_image, layers_, {L: 4 * sigma[L] for L in layers_}), "steer", a_, 4.0))
                arms.append((f"steer random, layers {a_} to {b_}", hooks.add_directions(UR, at_image, layers_, {L: 4 * sigma[L] for L in layers_}), "steer random", a_, 4.0))
        if args.mix and task in ("say", "animal"):
            for a in (0.25, 0.5, 0.75):
                arms += [(f"mix {a} at layer {L}", patch_mix([L], REC, pos, a), "mix", L, a) for L in sweep]
            arms += [(f"mix 0.5 norm-matched at layer {L}", patch_mix([L], REC, pos, 0.5, True), "mix norm", L, 0.5) for L in sweep]
        vocab_ids = {w: first_token_ids(lm, w, "word") for w in words} if kind == "word" and task == "say" else None
        for name, hk, kind_, L, alpha in arms:
            out = generate_scored(lm, rec_txt, rec_img, 5, hk, ids_x, ids_z, vocab_ids)
            for i, r in enumerate(S.itertuples()):
                gen, lpx, lpz, dist = out[i]
                res.append(
                    dict(
                        task=task,
                        needs=needs,
                        answer_type=kind,
                        arm=name,
                        kind=kind_,
                        layer=L,
                        alpha=alpha,
                        scene=r.scene,
                        X=r.X,
                        Z=r.Z,
                        q=r.q,
                        truth_X=TX[i],
                        truth_Z=TZ[i],
                        distinct=TX[i] != TZ[i],
                        generation=gen,
                        follows_X=matches(task, gen, TX[i], words),
                        keeps_Z=matches(task, gen, TZ[i], words),
                        lp_X=lpx,
                        lp_Z=lpz,
                        ld=lpx - lpz,
                        lp_vocab=(json.dumps(dist) if dist else None),
                    )
                )
            d_ = [x for x in res if x["task"] == task and x["arm"] == name and x["distinct"]]
            fx, kz, ldm = (np.mean([x[k] for x in d_]) for k in ("follows_X", "keeps_Z", "ld"))
            print(f"{task:9s} {name:34s} follows X {fx:.2f}  keeps Z {kz:.2f}  ld {ldm:+.2f}  (n {len(d_)})  [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(res)
    run.save_csv(R, "read_battery.csv")

    # ---- summary (rules frozen in RECORD.md, 21 Oct)
    out = dict(n=int(len(S)), n_layers=int(final + 1), sweep=sweep, questions={}, dropped={}, dose={})
    for task in tasks:
        d = R[(R.task == task) & R.distinct]
        scenes = np.sort(d.scene.unique())
        clean = d[d.kind == "clean"].set_index("scene").reindex(scenes)
        gate = float(clean.keeps_Z.mean())
        entry = dict(needs=QUESTIONS[task][1], answer_type=QUESTIONS[task][2], n=int(len(scenes)), clean_keeps_Z=gate, gate_pass=bool(gate >= 0.85), clean_ld=boot_scene(clean.ld.values, scenes))
        P = d[d.kind == "patch"]
        flips, lds = [], []
        for L in sweep:
            g = P[P.layer == L].set_index("scene").reindex(scenes)
            flips.append(boot_scene(g.follows_X.astype(float).values, scenes))
            lds.append(boot_scene(g.ld.values, scenes))
        entry["flip_by_layer"] = flips
        entry["ld_by_layer"] = lds
        entry["depth_flip"] = max([L for L, v in zip(sweep, flips) if v[0] >= 0.5], default=None)  # the last layer at which X's answer wins on half the scenes
        entry["depth_ld"] = max([L for L, v in zip(sweep, lds) if v[0] >= 0], default=None)  # the last layer at which X's answer is the more likely
        m = np.array([v[0] for v in lds])
        top, bottom = m[0], m[-1]
        if top - bottom > 1:  # transition width in layers: where the normalized log-probability difference passes 0.9 and 0.1 of its range
            z = (m - bottom) / (top - bottom)
            L90 = max([L for L, v in zip(sweep, z) if v >= 0.9], default=sweep[0])
            L10 = min([L for L, v in zip(sweep, z) if v <= 0.1 and L > L90], default=sweep[-1])
            entry["transition"] = dict(L90=L90, L10=L10, width=L10 - L90)
        out["questions"][task] = entry
        if not entry["gate_pass"]:
            out["dropped"][task] = gate
        if args.steer and task in ("say", "animal", "category", "set"):
            out.setdefault("steer", {})[task] = {}
            for kind_ in ("steer", "steer random"):
                for a_ in sorted(d[d.kind == kind_].layer.dropna().unique()):
                    g = d[(d.kind == kind_) & (d.layer == a_)].set_index("scene").reindex(scenes)
                    out["steer"][task][f"{kind_} from {int(a_)}"] = dict(
                        follows_X=boot_scene(g.follows_X.astype(float).values, scenes), keeps_Z=boot_scene(g.keeps_Z.astype(float).values, scenes), ld=boot_scene(g.ld.values, scenes)
                    )
        if args.mix and task in ("say", "animal"):
            out["dose"][task] = {}
            for kind_, a in (("mix", 0.25), ("mix", 0.5), ("mix", 0.75), ("mix norm", 0.5), ("patch", 1.0)):
                M = d[(d.kind == kind_) & (d.alpha == a)]
                curve = [boot_scene(M[M.layer == L].set_index("scene").reindex(scenes).ld.values, scenes) for L in sweep]
                out["dose"][task][f"{kind_} {a}"] = dict(ld_by_layer=curve, depth_ld=max([L for L, v in zip(sweep, curve) if v[0] >= 0], default=None), plateau=curve[0][0])
    print(
        "read depth by question (last layer where X's answer still wins; flips / log-prob): "
        + ", ".join(f"{t} {v['depth_flip']}/{v['depth_ld']}" + ("" if v["gate_pass"] else " [gate failed]") for t, v in out["questions"].items())
    )
    if out.get("steer"):
        print("steer: " + "; ".join(f"{t}: " + ", ".join(f"{k} follows X {v['follows_X'][0]:.2f}" for k, v in dd.items()) for t, dd in out["steer"].items()))
    if out["dose"]:
        print("dose-response depth_ld: " + "; ".join(f"{t}: " + ", ".join(f"{k} {v['depth_ld']}" for k, v in dd.items()) for t, dd in out["dose"].items()))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
