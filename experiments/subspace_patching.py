"""Subspace patching: the name without the fact (F4 in the notes).

A donor run sees object X in slot q of the six-slot canvas under a neutral prompt; a receiver sees object Z of the other kind
(animal or not, ridden or not) in the same slot and is asked about it. Parts of the donor's image-token states are patched into
the receiver: at the read layers the whole state, the lens subspace (the top r = 0.1 d directions of W_U J_L), its complement,
three random r-dimensional parts, J++ where fitted, and a placebo donor that saw the empty canvas; at every layer the whole state,
the lens part, a random part, and a donor with X in another slot; and the fact typed as text. Questions: say, letter, animal,
ride, and two joint questions (name and animal digit in one answer). Primary estimand: D = P(names X and gives Z's animal answer).
Runs on the fresh crop set (crop set 1). Optional rank sweep (--ranks).

Outputs in runs/subspace_patching/<model>/: subspace_patching.csv (task x arm x scene: answer, parsed noun and digit, carried,
kept), summary.json (with the COCO ids of every scene's crops), stimuli.png (donor and receiver canvases of scenes 0 and 1),
crops.json. Example 4, and the canvases of examples 1 to 4.
"""

import time

import numpy as np
import pandas as pd

from jspace import hooks, lens, stimuli
from jspace.analysis import boot_scene, fmt
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
INTRO = "Study the scene. It has an image grid with numbered slots."
TASKS = {
    "say": "What is the object in image slot {q}? Reply with the noun only.",
    "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
    "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
    "name then animal": "What is the object in image slot {q}, and is it an animal? Reply with the noun, then 7 if it is an animal or 3 if it is not.",
    "animal then name": "Is the object in image slot {q} an animal, and what is it? Reply with 7 if it is an animal or 3 if it is not, then the noun.",
}
RANDOM_SEEDS = [100, 400, 500]


def truth(task, x):
    if task == "say":
        return x
    if task == "letter":
        return x[0].upper()
    return "7" if x in (ANIMALS if task == "animal" else RIDDEN) else "3"


def main():
    ap = parser(__doc__, "subspace_patching")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--ranks", default="", help="exploratory rank sweep, e.g. 16,64,256,1024")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    assert args.crop_set == 1, "subspace patching runs on the fresh crops: --crop-set 1"
    ranks_sweep = [int(v) for v in args.ranks.split(",") if v.strip()]
    run = Run(args, "subspace_patching")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank = crops.words, crops.tiles
    layers, d_model = lm.layers, lm.d_model
    t0 = time.time()

    def parse(task, g):
        """(noun or None, digit or None) from an answer; single questions fill one field"""
        import re

        toks = re.findall(r"[a-z]+", g.lower())
        noun = next((w if w in words else w.rstrip("s") for w in toks if w in words or w.rstrip("s") in words), None)
        m = re.search(r"[73]", g)
        digit = m.group(0) if m else None
        if task == "say":
            return noun, None
        if task == "letter":
            return (g.strip().strip("*").upper()[:1] or None), None
        if task in ("animal", "ride"):
            return None, (g.strip().strip("*")[:1] if g.strip() else None)
        return noun, digit

    rng = np.random.default_rng(17)
    rows = []
    for i in range(args.n):
        pool = [w for w in words if (w in ANIMALS) == (i % 2 == 0)]
        x = pool[int(rng.integers(len(pool)))]
        zs = [w for w in words if (w in ANIMALS) != (x in ANIMALS) and (w in RIDDEN) != (x in RIDDEN) and w[0] != x[0]]
        assert zs, x
        z = zs[int(rng.integers(len(zs)))]
        q = int(rng.integers(1, stimuli.N_SLOTS + 1))
        q2 = int(rng.choice([s for s in range(1, stimuli.N_SLOTS + 1) if s != q]))
        jx, jz = i % len(bank[x]), (i + 1) % len(bank[z])
        rows.append(dict(scene=i, X=x, Z=z, q=q, q2=q2, tx=bank[x][jx], tz=bank[z][jz], crop_x=crops.coco[x][jx], crop_z=crops.coco[z][jz]))
    S = pd.DataFrame(rows)
    don_img = [stimuli.scene(r.q, r.tx) for r in S.itertuples()]
    don2_img = [stimuli.scene(r.q2, r.tx) for r in S.itertuples()]
    rec_img = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
    blank_img = [stimuli.scene(None, None)] * len(S)
    don_txt = [lm.chat_media([INTRO, "IMAGE"])] * len(S)
    ids0 = lm.encode(don_txt[:1], don_img[:1])["input_ids"][0].tolist()
    img_pos = [j for j, t in enumerate(ids0) if t == lm.image_token_id]  # every image token, whatever the family's grid
    n_img = len(img_pos)
    vs_off = img_pos[0]
    LF = lm.all_layers()
    R = int(0.1 * d_model)
    Qs = {L: lens.lens_subspace(lm, L, R) for L in layers}
    QsA = {L: (lens.lens_subspace(lm, L, R) if 0 <= L < lm.final else Qs[layers[0] if L < 0 else layers[-1]]) for L in LF}
    QrA = {L: lens.random_subspace(d_model, R, 100 + L) for L in LF}
    Qr3 = {seed: {L: lens.random_subspace(d_model, R, seed + L) for L in layers} for seed in RANDOM_SEEDS}
    Qpp = {L: lens.jpp_subspace(lm, L, R) for L in layers} if lm.jpp is not None else None
    print(f"rank r = {R} (0.1 d); crops from set {args.crop_set}; {len(S)} scenes  [{time.time() - t0:.0f}s]")

    def capture(images):
        rec = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
        for j in range(len(S)):
            _, Rr = lm.forward(don_txt[j : j + 1], images[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
            for L in LF:
                rec[L][j] = Rr[L][0].half().numpy()
        return rec

    REC, REC2, RECB = capture(don_img), capture(don2_img), capture(blank_img)
    print(f"donor states captured ({len(LF)} layers)  [{time.time() - t0:.0f}s]")
    stimuli.contact_sheet(
        [(f"donor: {r.X} in slot {r.q}", don_img[i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"receiver: {r.Z} in slot {r.q}", rec_img[i]) for i, r in enumerate(S.head(2).itertuples())],
        4,
        run.path("stimuli.png"),
    )
    res = []
    for task, tmpl in TASKS.items():
        rec_txt = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q)]) for r in S.itertuples()]
        typed = [lm.chat([("user", INTRO + f" Image slot {r.q} contains a {r.X}. " + tmpl.format(q=r.q))]) for r in S.itertuples()]
        n_all = [len(lm.encode(rec_txt[j : j + 1], [rec_img[j]])["input_ids"][0]) for j in range(len(S))]
        first = [j for j, t in enumerate(lm.encode(rec_txt[:1], rec_img[:1])["input_ids"][0].tolist()) if t == lm.image_token_id][0]
        assert first == vs_off, "the image does not sit at the donor's positions"
        pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
        arms = [("receiver alone", None), ("full state, read layers", hooks.patch(layers, REC, pos)), ("lens part, read layers", hooks.patch(layers, REC, pos, Qs, "lens"))]
        if Qpp is not None:
            arms.append(("J++ part, read layers", hooks.patch(layers, REC, pos, Qpp, "lens")))
        arms += [(f"random part, read layers, draw {k + 1}", hooks.patch(layers, REC, pos, Qr3[seed], "lens")) for k, seed in enumerate(RANDOM_SEEDS)]
        arms += [
            ("complement of the lens part, read layers", hooks.patch(layers, REC, pos, Qs, "complement")),
            ("placebo donor (empty canvas), read layers", hooks.patch(layers, RECB, pos)),
            ("full state, every layer", hooks.patch(LF, REC, pos)),
            ("lens part, every layer", hooks.patch(LF, REC, pos, QsA, "lens")),
            ("random part, every layer", hooks.patch(LF, REC, pos, QrA, "lens")),
            ("non-target donor, full state, every layer", hooks.patch(LF, REC2, pos)),
            ("the fact typed as text", "typed"),
        ]
        if ranks_sweep and task in ("say", "animal"):
            for rk in ranks_sweep:
                if rk == R:
                    continue
                arms += [
                    (f"lens part r={rk}, read layers", hooks.patch(layers, REC, pos, {L: lens.lens_subspace(lm, L, rk) for L in layers}, "lens")),
                    (f"random part r={rk}, read layers", hooks.patch(layers, REC, pos, {L: lens.random_subspace(d_model, rk, 700 + L) for L in layers}, "lens")),
                ]
        for name, hk in arms:
            tx, im, hk = (typed, [None] * len(S), None) if isinstance(hk, str) else (rec_txt, rec_img, hk)
            gen = lm.generate(tx, im, 8, hk)
            for i, r in enumerate(S.itertuples()):
                noun, digit = parse(task, gen[i])
                tX, tZ = truth(task, r.X), truth(task, r.Z)
                if task == "say":
                    carried, kept = noun == r.X, noun == r.Z
                elif task == "letter":
                    carried, kept = noun == tX, noun == tZ
                elif task in ("animal", "ride"):
                    carried, kept = digit == tX, digit == tZ
                else:
                    carried, kept = (noun == r.X and digit == truth("animal", r.X)), (noun == r.Z and digit == truth("animal", r.Z))
                joint = task.startswith(("name", "animal then"))
                res.append(
                    dict(
                        task=task,
                        arm=name,
                        scene=r.scene,
                        X=r.X,
                        Z=r.Z,
                        q=r.q,
                        generation=gen[i],
                        noun=noun,
                        digit=digit,
                        carried=bool(carried),
                        kept=bool(kept),
                        name_X=(noun == r.X),
                        digit_Z=(digit == truth("animal", r.Z)) if joint else None,
                    )
                )
            d_ = [x for x in res if x["task"] == task and x["arm"] == name]
            print(
                f"{task:16s} {name:44s} carried {np.mean([x['carried'] for x in d_]):.2f}  kept {np.mean([x['kept'] for x in d_]):.2f}"
                + (f"  name X & digit Z {np.mean([bool(x['name_X'] and x['digit_Z']) for x in d_]):.2f}" if task.startswith(("name", "animal then")) else "")
                + f"  [{time.time() - t0:.0f}s]"
            )
    F4 = pd.DataFrame(res)
    run.save_csv(F4, "subspace_patching.csv")

    # ---- summary (rules: RECORD.md 04:45)
    def vals(arm, task, col="carried"):
        return F4[(F4.arm == arm) & (F4.task == task)].set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values

    ARMS = [a for a in F4.arm.unique() if "r=" not in a]

    def boot(v):
        return boot_scene(v, S.scene.values)

    out = dict(n=int(len(S)), rank=R, crop_set=args.crop_set, crops=S[["scene", "X", "Z", "q", "crop_x", "crop_z"]].to_dict("records"), carried={}, kept={}, D={}, D_ride={}, joint={}, verdict={})
    for task in TASKS:
        out["carried"][task] = {arm: boot(vals(arm, task)) for arm in ARMS}
        out["kept"][task] = {arm: boot(vals(arm, task, "kept")) for arm in ARMS}
    for arm in ARMS:
        out["D"][arm] = boot(vals(arm, "say") * vals(arm, "animal", "kept"))
        out["D_ride"][arm] = boot(vals(arm, "say") * vals(arm, "ride", "kept"))
        out["joint"][arm] = {}
        for task in ("name then animal", "animal then name"):
            d = F4[(F4.arm == arm) & (F4.task == task)].set_index("scene")
            out["joint"][arm][task] = boot((d.name_X.astype(bool) & d.digit_Z.astype(bool)).astype(float).reindex(S.scene).fillna(0).values)
    rnd_say = np.mean([vals(f"random part, read layers, draw {k + 1}", "say") for k in range(3)], 0)
    rnd_D = np.mean([vals(f"random part, read layers, draw {k + 1}", "say") * vals(f"random part, read layers, draw {k + 1}", "animal", "kept") for k in range(3)], 0)
    out["D"]["random part, read layers (mean of 3)"] = boot(rnd_D)
    gate = {t: out["kept"][t]["receiver alone"][0] >= 0.85 for t in ("animal", "ride")}
    pc = dict(
        full_read_names_X=out["carried"]["say"]["full state, read layers"][0] >= 0.8,
        full_every_flips_animal=out["carried"]["animal"]["full state, every layer"][0] >= 0.8,
        placebo_inert=(out["carried"]["say"]["placebo donor (empty canvas), read layers"][0] <= 0.1 and out["kept"]["animal"]["placebo donor (empty canvas), read layers"][0] >= 0.85),
    )
    dl = boot(vals("lens part, read layers", "say") * vals("lens part, read layers", "animal", "kept") - vals("full state, read layers", "say") * vals("full state, read layers", "animal", "kept"))
    ls = boot(vals("lens part, read layers", "say") - rnd_say)
    out["lens_minus_full_D"], out["lens_minus_random_say"], out["gate"], out["positive_controls"] = dl, ls, gate, pc
    if not (gate["animal"] and pc["full_read_names_X"] and pc["full_every_flips_animal"] and pc["placebo_inert"]):
        head = "positive control or gate failed: " + ", ".join(k for k, v in {**gate, **pc}.items() if not v)
    elif dl[0] >= 0.4 and dl[1] > 0 and ls[0] >= 0.3 and ls[1] > 0:
        head = "name without fact: the lens part installs the donor's name and leaves the receiver's fact in place"
    else:
        head = "no dissociation at this n"
    out["verdict"]["headline"] = head
    print("D = P(names X and the animal answer is Z's): " + "  ".join(f"{a}: {fmt(v)}" for a, v in out["D"].items() if "draw" not in a))
    print(f"gate {gate}; positive controls {pc}; lens - full on D {fmt(dl)}; lens - random on say {fmt(ls)} -> {head}")
    if ranks_sweep:
        out["rank_sweep"] = {arm: {t: boot(vals(arm, t)) for t in ("say", "animal")} for arm in F4.arm.unique() if "r=" in arm}
        print("rank sweep (say / animal carried): " + "  ".join(f"{a}: {v['say'][0]:.2f} / {v['animal'][0]:.2f}" for a, v in out["rank_sweep"].items()))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
