"""Back-patching: the same state is used or ignored depending only on where it is patched in (D2 in the notes).

The depth sweep's scenes and its two cutoffs for this model, read from runs/depth_sweep/<model>/summary.json: L_use, the last
window where the animal rule follows the donor, and L_say, the last where naming does. "Early" is every layer up to L_use + 3,
"the band" the layers between the two. Arms on the image tokens: the donor's state in the early layers only; the same with the
receiver's own clean state restored in the band (the restore arm); the donor's state in the band only; early and band; every layer;
and cross-layer patching: the donor's states from the first 4 band layers back-patched into the last 4 early layers, the donor's
last 4 early states patched into the first 4 band layers, and the early-window states in place (the control). The same for a
four-frame video donor (the canvas as four frames). Also: is the category decodable from the donor's state at every depth
(leave-one-out nearest centroid)? Questions: say, animal.

Outputs in runs/back_patching/<model>/: back_patching.csv (source x task x arm x scene: answer, follows X, keeps Z), summary.json,
crops.json. Example 3.
"""

import json
import time

import numpy as np
import pandas as pd

from jspace import hooks, stimuli
from jspace.analysis import boot_scene, noun_matches
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
INTRO = "Study the scene. It has an image grid with numbered slots."
TASKS = {
    "say": "What is the object in image slot {q}? Reply with the noun only.",
    "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
}


def main():
    ap = parser(__doc__, "back_patching")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--frames", type=int, default=4, help="frames of the video donor (1: no video donor)")
    ap.add_argument("--horizons", default=None, help="the depth sweep's summary.json (default runs/depth_sweep/<model>/summary.json)")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    tag = args.model.split("/")[-1]
    horizons = args.horizons or args.runs_dir / "depth_sweep" / tag / "summary.json"
    with open(horizons) as f:  # fails here, before loading the model, when the depth sweep has not run on this model
        cut = json.load(f)["cut"]["transplant"]
    L_use, L_say = cut["L_use"], cut["L_say"]
    assert L_use is not None and L_say is not None, cut
    run = Run(args, "back_patching")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank = crops.words, crops.tiles
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
        rows.append(dict(scene=i, X=x, Z=z, q=q, tx=bank[x][i % len(bank[x])], tz=bank[z][(i + 1) % len(bank[z])]))
    S = pd.DataFrame(rows)
    don_img = [stimuli.scene(r.q, r.tx) for r in S.itertuples()]
    rec_img = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
    final, d_model = lm.final, lm.d_model
    EARLY = [L for L in range(-1, final + 1) if L <= L_use + 3]
    BAND = [L for L in range(-1, final + 1) if L_use + 3 < L <= L_say + 3]
    LATE = [L for L in range(-1, final + 1) if L > L_say + 3]
    print(f"horizons from the depth sweep: L_use {L_use}, L_say {L_say}; early {EARLY[0]}..{EARLY[-1]}, band {BAND[0]}..{BAND[-1]}, late {LATE[0] if LATE else None}..{final}")
    LF = lm.all_layers()
    sources = {"picture": (don_img, rec_img, [lm.chat_media([INTRO, "IMAGE"])] * len(S), lambda task, r: lm.chat_media([INTRO, "IMAGE", TASKS[task].format(q=r.q)]))}
    if args.frames > 1:
        k = args.frames
        sources["video"] = (
            [[im] * k for im in don_img],
            [[im] * k for im in rec_img],
            [lm.chat_media([INTRO] + ["IMAGE"] * k)] * len(S),
            lambda task, r: lm.chat_media([INTRO] + ["IMAGE"] * k + [TASKS[task].format(q=r.q)]),
        )
    res, decodable = [], {}
    for src, (dimg, rimg, dtxt, rprompt) in sources.items():
        try:
            ids0 = lm.encode(dtxt[:1], dimg[:1])["input_ids"][0].tolist()
        except Exception as e:  # noqa: BLE001 - a processor without video input skips that donor
            print(f"{src}: the processor does not take this input ({type(e).__name__}: {str(e)[:120]}); skipped")
            continue
        img_pos = [j for j, t in enumerate(ids0) if t == lm.image_token_id]
        n_img = len(img_pos)
        REC = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
        RECZ = {L: np.zeros((len(S), n_img, d_model), np.float16) for L in LF}
        for j in range(len(S)):
            _, Rr = lm.forward(dtxt[j : j + 1], dimg[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
            _, Rz = lm.forward(dtxt[j : j + 1], rimg[j : j + 1], LF, where=lambda ids, _t=img_pos: list(_t))
            for L in LF:
                REC[L][j] = Rr[L][0].half().numpy()
                RECZ[L][j] = Rz[L][0].half().numpy()
        print(f"{src}: {n_img} image tokens, states captured  [{time.time() - t0:.0f}s]")
        # is the category decodable from the donor's state at every depth? leave-one-out nearest centroid on the slot's mean state
        slot_idx = {s: stimuli.slot_tokens(stimuli.COLS, stimuli.ROWS, s) for s in range(1, stimuli.N_SLOTS + 1)} if n_img == stimuli.N_SLOTS * 49 else None
        lab = np.array([r.X in ANIMALS for r in S.itertuples()])
        dec = {}
        for L in LF:
            F = np.stack([REC[L][i][slot_idx[r.q] if slot_idx else slice(None)].astype(np.float32).mean(0) for i, r in enumerate(S.itertuples())])
            F = (F - F.mean(0)) / (F.std(0) + 1e-6)
            ok = []
            for i in range(len(S)):
                m = np.ones(len(S), bool)
                m[i] = False
                c1, c0 = F[m & lab].mean(0), F[m & ~lab].mean(0)
                ok.append((np.linalg.norm(F[i] - c1) < np.linalg.norm(F[i] - c0)) == lab[i])
            dec[int(L)] = float(np.mean(ok))
        decodable = dec  # as in the original runs, the summary keeps the last donor's
        print(f"{src}: category decodable from the donor's state (leave-one-out, by layer): " + " ".join(f"{L}:{v:.2f}" for L, v in dec.items()))
        # the same late state delivered early, and the early state delivered late
        E4, B4 = EARLY[-4:], BAND[:4]
        rec_late_early = {Lt: REC[Ls] for Lt, Ls in zip(E4, B4)}
        rec_early_late = {Lt: REC[Ls] for Lt, Ls in zip(B4, E4)}
        for task in TASKS:
            rec_txt = [rprompt(task, r) for r in S.itertuples()]
            n_all = [len(lm.encode(rec_txt[j : j + 1], [rimg[j]])["input_ids"][0]) for j in range(len(S))]
            pos = [(lambda s, _n=n_all[j]: np.array([s - _n + t for t in img_pos])) for j in range(len(S))]
            arms = [
                ("receiver alone", None),
                ("donor early only", hooks.patch(EARLY, REC, pos)),
                ("donor early, receiver restored in the band", hooks.combine(hooks.patch(EARLY, REC, pos), hooks.patch(BAND, RECZ, pos))),
                ("donor in the band only", hooks.patch(BAND, REC, pos)),
                ("donor early and band", hooks.patch(EARLY + BAND, REC, pos)),
                ("donor every layer", hooks.patch(LF, REC, pos)),
                ("donor's band state delivered in the early window", hooks.patch(E4, rec_late_early, pos)),
                ("donor's early state delivered in the band", hooks.patch(B4, rec_early_late, pos)),
                ("donor's early-window state in place (control)", hooks.patch(E4, REC, pos)),
            ]
            for name, hk in arms:
                gen = lm.generate(rec_txt, rimg, 5, hk)
                for i, r in enumerate(S.itertuples()):
                    tX = r.X if task == "say" else ("7" if r.X in ANIMALS else "3")
                    tZ = r.Z if task == "say" else ("7" if r.Z in ANIMALS else "3")
                    fx = bool(noun_matches([gen[i]], [tX], set(words))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tX
                    kz = bool(noun_matches([gen[i]], [tZ], set(words))[0]) if task == "say" else gen[i].strip().strip("*")[:1] == tZ
                    res.append(dict(source=src, task=task, arm=name, scene=r.scene, X=r.X, Z=r.Z, generation=gen[i], follows_X=fx, keeps_Z=kz))
                d_ = [x for x in res if x["source"] == src and x["task"] == task and x["arm"] == name]
                print(f"{src:7s} {task:6s} {name:44s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  keeps Z {np.mean([x['keeps_Z'] for x in d_]):.2f}  [{time.time() - t0:.0f}s]")
        del REC, RECZ
    D2 = pd.DataFrame(res)
    run.save_csv(D2, "back_patching.csv")

    # ---- summary (rules: RECORD.md 05:45 and 06:05)
    def boot(v):
        return boot_scene(v, S.scene.values)

    out = dict(n=int(len(S)), L_use=L_use, L_say=L_say, early=[EARLY[0], EARLY[-1]], band=[BAND[0], BAND[-1]], decodable_by_layer=decodable, arms={}, verdict={})
    for src in D2.source.unique():
        out["arms"][src] = {}
        for arm in D2[D2.source == src].arm.unique():

            def g(task, col, src=src, arm=arm):
                return D2[(D2.source == src) & (D2.task == task) & (D2.arm == arm)].set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values

            out["arms"][src][arm] = dict(
                say_X=boot(g("say", "follows_X")),
                say_Z=boot(g("say", "keeps_Z")),
                animal_X=boot(g("animal", "follows_X")),
                animal_Z=boot(g("animal", "keeps_Z")),
                says_Z_and_classifies_X=boot(g("say", "keeps_Z") * g("animal", "follows_X")),
                says_X_and_classifies_Z=boot(g("say", "follows_X") * g("animal", "keeps_Z")),
            )
        r_, b_ = out["arms"][src]["donor early, receiver restored in the band"], out["arms"][src]["donor in the band only"]
        out["verdict"][src] = (
            "two horizons, in both directions"
            if r_["say_Z"][0] >= 0.6 and r_["animal_X"][0] >= 0.6 and b_["say_X"][0] >= 0.6 and b_["animal_Z"][0] >= 0.6
            else "one direction only"
            if b_["say_X"][0] >= 0.6 and b_["animal_Z"][0] >= 0.6
            else "not shown"
        )
        print(
            f"{src}: restore arm says Z {r_['say_Z'][0]:.2f}, classifies X {r_['animal_X'][0]:.2f}; "
            f"band-only arm says X {b_['say_X'][0]:.2f}, classifies Z {b_['animal_Z'][0]:.2f} -> {out['verdict'][src]}"
        )
        t_ = out["arms"][src].get("donor's band state delivered in the early window")
        u_ = out["arms"][src].get("donor's early-window state in place (control)")
        if t_ and u_:
            out["verdict"][f"{src} retimed"] = (
                "the late state still holds the fact: delivered early it is used" if t_["animal_X"][0] >= 0.6 else "the late state has lost the fact" if t_["animal_X"][0] <= 0.2 else "undetermined"
            )
            print(f"{src}: the band state delivered early classifies X {t_['animal_X'][0]:.2f} (the early state in place: {u_['animal_X'][0]:.2f}) -> {out['verdict'][f'{src} retimed']}")
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
