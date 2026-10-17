"""Depth sweep: in which layers can a seen object still be named, and in which still used? (D1 in the notes)

A donor run sees object X in slot q of the six-slot canvas (no question); a receiver sees Z, of the other kind, in the same slot.
The donor's image-token states are patched into the receiver at one window of 4 consecutive layers at a time, sliding from the
input (layer -1) to the last layer; the same with a printed-word donor (X's name printed in the slot) and a typed-sentence donor
("Image slot q contains a X.", only the state of the word X, patched into the position of Z's name). Also per window: Z's lens
direction steered in at 4 sigma_L (and a random direction); and at the read layers the lens direction of the word "animal".
Questions: say and the animal rule; with --spectrum also the first letter, membership in a random set of three names, and the
ride rule. The last window at which an answer still follows the donor gives each question's horizon.

Outputs in runs/<run>/<model>/: depth_sweep.csv (task x arm x scene: answer, follows X, keeps Z), summary.json (curves, horizons,
the two cutoffs L_say and L_use that back_patching reads), crops.json. Examples 1 and 2 (example 1 from the --spectrum run).
"""

import time

import numpy as np
import pandas as pd

from jspace import hooks, lens, stimuli
from jspace.analysis import boot_scene, noun_matches
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
WINDOW = 4
INTRO = "Study the scene. It has an image grid with numbered slots."
TASKS = {
    "say": "What is the object in image slot {q}? Reply with the noun only.",
    "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
}
SPECTRUM = {  # a lexical, a symbolic and a functional property beside say and animal (amendment of 14 Oct 05:30)
    "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
    "set": "Rule: if the object in image slot {q} is a {a}, a {b} or a {c}, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
}


def word_positions(lm, prompt, word, occurrence_after):
    """token indices of `word` in a text-only prompt, its first occurrence after character occurrence_after"""
    enc = lm.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    i = prompt.index(word, occurrence_after)
    return [t for t, (a, b) in enumerate(enc["offset_mapping"]) if a < i + len(word) and b > i]


def main():
    ap = parser(__doc__, "depth_sweep")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--spectrum", action="store_true", help="add the letter, set and ride questions")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "depth_sweep")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank, sid = crops.words, crops.tiles, crops.ids
    tasks = {**TASKS, **SPECTRUM} if args.spectrum else dict(TASKS)
    t0 = time.time()

    rng = np.random.default_rng(17)
    rows = []
    for i in range(args.n):
        pool = [w for w in words if (w in ANIMALS) == (i % 2 == 0)]
        x = pool[int(rng.integers(len(pool)))]
        zs = [w for w in words if (w in ANIMALS) != (x in ANIMALS) and (w in RIDDEN) != (x in RIDDEN) and w[0] != x[0]]
        z = zs[int(rng.integers(len(zs)))]
        q = int(rng.integers(1, stimuli.N_SLOTS + 1))
        rng.choice([s for s in range(1, stimuli.N_SLOTS + 1) if s != q])  # a second slot, unused here: drawn to keep the scene sequence
        others = [w for w in words if w not in (x, z)]
        fill = [others[j] for j in rng.choice(len(others), 2, replace=False)]
        sset = [x if i % 4 < 2 else z] + fill
        rng.shuffle(sset)
        rows.append(dict(scene=i, X=x, Z=z, q=q, set=sset, tx=bank[x][i % len(bank[x])], tz=bank[z][(i + 1) % len(bank[z])]))
    S = pd.DataFrame(rows)
    don_img = [stimuli.scene(r.q, r.tx) for r in S.itertuples()]
    prt_img = [stimuli.scene(r.q, stimuli.word_tile(r.X)) for r in S.itertuples()]  # the same fact as a printed word
    rec_img = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
    don_txt = [lm.chat_media([INTRO, "IMAGE"])] * len(S)
    don_typed = [lm.chat([("user", INTRO + f" Image slot {r.q} contains a {r.X}.")]) for r in S.itertuples()]
    pos_typed = [word_positions(lm, t, r.X, t.index("contains a "))[:1] for t, r in zip(don_typed, S.itertuples())]
    ids0 = lm.encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
    img_pos = [j for j, t in enumerate(ids0) if t == lm.image_token_id]
    n_img = len(img_pos)
    LF = lm.all_layers()
    final, d_model = lm.final, lm.d_model
    REC = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
    RECZ = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
    RECP = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
    RECT = {L: np.zeros((len(S), 1, d_model), np.float16) for L in LF}
    for j in range(len(S)):
        _, Rt = lm.forward(don_typed[j : j + 1], None, LF, where=lambda ids, _t=pos_typed[j]: list(_t))
        for L in LF:
            RECT[L][j] = Rt[L][0].half().numpy()
        _, Rr = lm.forward(don_txt[j : j + 1], don_img[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        _, Rz = lm.forward(don_txt[j : j + 1], rec_img[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        _, Rp = lm.forward(don_txt[j : j + 1], prt_img[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
        for L in LF:
            REC[L][j] = Rr[L][0].half().numpy()
            RECZ[L][j] = Rz[L][0].half().numpy()
            RECP[L][j] = Rp[L][0].half().numpy()
    print(f"states captured ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
    WINDOWS = [list(range(a, min(a + WINDOW, final + 1))) for a in range(-1, final + 1, WINDOW)]
    ids_x = [sid[w][0] for w in S.X]
    ids_y = [sid[w][0] for w in S.Z]  # the steering word: the receiver's own object's name
    ids_animal = lm.single_ids(["animal", " animal", "Animal", " Animal"])
    nearest = lambda L: L if (0 <= L < final and L in lm.jac) else (0 if L < 0 else final - 1)  # noqa: E731 - the nearest fitted map
    UY = {L: lens.word_directions(lm, nearest(L), ids_y, "J") for L in LF}
    UX = {L: lens.word_directions(lm, nearest(L), ids_x, "J") for L in LF}
    UA = {L: lens.word_directions(lm, nearest(L), [ids_animal[0]] * len(S), "J") for L in LF}
    UR = {L: lens.unit_rows(len(S), d_model, 900 + L) for L in LF}
    slot_idx = {s: stimuli.slot_tokens(stimuli.COLS, stimuli.ROWS, s) for s in range(1, stimuli.N_SLOTS + 1)} if n_img == stimuli.N_SLOTS * 49 else None
    sigma = {L: float(np.mean([np.abs(RECZ[L][i][slot_idx[r.q] if slot_idx else slice(None)].astype(np.float32) @ UX[L][i, :, 0]).mean() for i, r in enumerate(S.itertuples())])) for L in LF}
    at_image = hooks.image_positions(lm.image_token_id)

    def steer(U, layers, scale):
        return hooks.add_directions(U, at_image, layers, scale)

    def truth(t, r, w):
        if t == "say":
            return w
        if t == "letter":
            return w[0].upper()
        if t == "set":
            return "7" if w in r.set else "3"
        return "7" if w in (RIDDEN if t == "ride" else ANIMALS) else "3"

    res = []
    TX = {t: [truth(t, r, r.X) for r in S.itertuples()] for t in tasks}
    TZ = {t: [truth(t, r, r.Z) for r in S.itertuples()] for t in tasks}
    for task, tmpl in tasks.items():
        rec_txt = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set[0], b=r.set[1], c=r.set[2])]) for r in S.itertuples()]
        n_all = [len(lm.encode(rec_txt[j : j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
        pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
        arms = [("clean", None, "none", 0)]
        for w in WINDOWS:
            arms += [
                (f"move the donor's state, layers {w[0]} to {w[-1]}", hooks.patch(w, REC, pos), "transplant", w[0]),
                (f"move the printed-word donor's state, layers {w[0]} to {w[-1]}", hooks.patch(w, RECP, pos), "transplant printed", w[0]),
                (f"inject Y, layers {w[0]} to {w[-1]}", steer(UY, w, {L: 4 * sigma[L] for L in w}), "inject", w[0]),
                (f"inject random, layers {w[0]} to {w[-1]}", steer(UR, w, {L: 4 * sigma[L] for L in w}), "random", w[0]),
            ]
        arms += [(f"inject the word 'animal' x{a}, read layers", steer(UA, lm.layers, {L: a * sigma[L] for L in lm.layers}), "C1 animal", a) for a in (4, 8)]
        arms += [(f"inject random x{a}, read layers", steer(UR, lm.layers, {L: a * sigma[L] for L in lm.layers}), "C1 random", a) for a in (4, 8)]
        rec_typed = [lm.chat([("user", INTRO + f" Image slot {r.q} contains a {r.Z}. " + tmpl.format(q=r.q, a=r.set[0], b=r.set[1], c=r.set[2]))]) for r in S.itertuples()]
        n_typed = [len(lm.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in rec_typed]
        pos_t = [(lambda s, _n=n_typed[j], _p=pos_typed[j]: np.array([s - _n + t for t in _p])) for j in range(len(S))]
        arms.append(("typed: receiver alone", None, "typed none", 0))
        arms += [(f"move the typed donor's state, layers {w[0]} to {w[-1]}", hooks.patch(w, RECT, pos_t), "transplant typed", w[0]) for w in WINDOWS]
        for name, hk, kind, w0 in arms:
            typed_arm = kind.startswith("typed") or kind == "transplant typed"
            gen = lm.generate(rec_typed if typed_arm else rec_txt, [None] * len(S) if typed_arm else rec_img, 5, hk)
            for i, r in enumerate(S.itertuples()):
                if task == "say":
                    ok_x = bool(noun_matches([gen[i]], [TX[task][i]], set(words))[0])
                    ok_z = bool(noun_matches([gen[i]], [TZ[task][i]], set(words))[0])
                else:
                    ok_x = gen[i].strip().strip("*").upper()[:1] == TX[task][i]
                    ok_z = gen[i].strip().strip("*").upper()[:1] == TZ[task][i]
                res.append(dict(task=task, arm=name, kind=kind, w0=w0, scene=r.scene, X=r.X, Z=r.Z, generation=gen[i], follows_X=ok_x, keeps_Z=ok_z))
            d_ = [x for x in res if x["task"] == task and x["arm"] == name]
            print(f"{task:6s} {name:46s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  keeps Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  [{time.time() - t0:.0f}s]")
    D1 = pd.DataFrame(res)
    run.save_csv(D1, "depth_sweep.csv")

    # ---- summary (rules: RECORD.md 04:55 and 05:05)
    def boot(v):
        return boot_scene(v, S.scene.values)

    def val(arm, task):
        return D1[(D1.arm == arm) & (D1.task == task)].set_index("scene").follows_X.astype(float).reindex(S.scene).fillna(0).values

    out = dict(n=int(len(S)), n_layers=int(final + 1), windows=[w[0] for w in WINDOWS], sigma=sigma, curves={}, C1={}, cut={})
    arm_order = D1[["arm", "kind", "w0"]].drop_duplicates().sort_values("w0").to_dict("records")
    for kind in ("transplant", "transplant printed", "transplant typed", "inject", "random"):
        out["curves"][kind] = {task: [boot(val(a["arm"], task)) for a in arm_order if a["kind"] == kind] for task in tasks}
    out["horizon"] = {}  # the last window start at which each question follows the donor on >= 0.5 of scenes, per data type
    for kind in ("transplant", "transplant printed", "transplant typed"):
        out["horizon"][kind] = {task: max([w for w, v in zip(out["windows"], out["curves"][kind][task]) if v[0] >= 0.5], default=None) for task in tasks}
    print("horizons (last window start with >= 0.5): " + "; ".join(f"{k}: " + ", ".join(f"{t} {v}" for t, v in h.items()) for k, h in out["horizon"].items()))
    for kind in ("transplant", "transplant printed", "transplant typed", "inject"):
        cs, ca = out["curves"][kind]["say"], out["curves"][kind]["animal"]
        rnd = out["curves"]["random"]["animal"]
        say_ok = [w for w, v in zip(out["windows"], cs) if v[0] >= 0.5]
        use_ok = [w for w, v, rv in zip(out["windows"], ca, rnd) if v[0] >= rv[0] + 0.3]
        out["cut"][kind] = dict(
            L_say=(max(say_ok) if say_ok else None),
            L_use=(max(use_ok) if use_ok else None),
            say_windows=say_ok,
            use_windows=use_ok,
            verdict=("never" if say_ok and not use_ok else "too late" if say_ok and use_ok and max(use_ok) <= max(say_ok) - 2 * WINDOW else "together" if say_ok and use_ok else "nothing moves"),
        )
    for a in (4, 8):
        fa, fr = val(f"inject the word 'animal' x{a}, read layers", "animal"), val(f"inject random x{a}, read layers", "animal")
        keeps = D1[(D1.arm == f"inject the word 'animal' x{a}, read layers") & (D1.task == "say")].set_index("scene").keeps_Z.astype(float).reindex(S.scene).fillna(0).values
        c1 = out["C1"][a] = dict(animal_follows=boot(fa), random=boot(fr), diff=boot(fa - fr), say_keeps_Z=boot(keeps))
        c1["verdict"] = "the rule reads verbal category content" if c1["diff"][0] >= 0.3 and c1["diff"][1] > 0 else "the rule reads non-verbal features" if c1["diff"][0] <= 0.1 else "undetermined"
    print("depth: " + "; ".join(f"{k}: L_say {v['L_say']} L_use {v['L_use']} -> {v['verdict']}" for k, v in out["cut"].items()))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
