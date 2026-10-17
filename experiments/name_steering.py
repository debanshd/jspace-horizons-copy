"""Name steering: does the model use what it sees through the direction that lets it say it? (G1 in the notes)

One COCO crop X in slot q of the six-slot canvas. Four questions about slot q: say (the noun), letter (its first letter), animal
(7 / 3), ride (7 / 3: something people ride or drive). Removal on the slot's 49 image tokens at the read layers (mean-ablated):
the own name's lens direction (rank 1), another name Y's, a random direction, the own name's plain unembedding direction, J++
where fitted, the word "animal"'s direction, the own name orthogonal to "animal", the slot's top principal direction (the energy
control), the 256-dimensional lens subspace, a random 256-dimensional part; and the printed word (X's name in the slot) clean and
with the own name removed. Steering: alpha * sigma_L along Y (the other kind), Z (the same kind), a random direction and Y's plain
unembedding direction, at three sites (the slot's tokens, every image token, the last prompt token), alpha in {1, 2, 4, 8}; the
dose alpha* per site is picked on the first 16 scenes and the verdicts are read on the other 48. Steering along X on the empty canvas.

Outputs in runs/name_steering/<model>/: name_steering.csv (block x arm x alpha x task x scene: the answer, correct, follows the
steering word), logp.csv (log p of the correct first token under each removal), summary.json, stimuli.png, crops.json. Example 5.
"""

import time

import numpy as np
import pandas as pd
import torch

from jspace import hooks, lens, stimuli
from jspace.analysis import boot_scene, fmt, noun_matches
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

INTRO = "Study the scene. It has an image grid with numbered slots."
TASKS = {
    "say": "What is the object in image slot {q}? Reply with the noun only.",
    "letter": "What is the first letter of the name of the object in image slot {q}? Reply with the letter only.",
    "animal": "Rule: if the object in image slot {q} is an animal, answer 7; otherwise answer 3. Reply with the number only.",
    "ride": "Rule: if the object in image slot {q} is something people ride or drive, answer 7; otherwise answer 3. Reply with the number only.",
}
RULES = ["animal", "ride"]
ALPHAS = [1.0, 2.0, 4.0, 8.0]
ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN


def truth(task, x):
    if task == "say":
        return x
    if task == "letter":
        return x[0].upper()
    return "7" if x in (ANIMALS if task == "animal" else RIDDEN) else "3"


def main():
    ap = parser(__doc__, "name_steering")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--n-cal", type=int, default=16, help="calibration scenes (the dose is picked on them)")
    ap.add_argument("--crop-set", type=int, default=0)
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "name_steering")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank, sid = crops.words, crops.tiles, crops.ids
    layers, d_model, image_id = lm.layers, lm.d_model, lm.image_token_id
    t0 = time.time()

    def score(task, gens, truths):
        if task == "say":
            return [bool(v) for v in noun_matches(gens, truths, set(words))]
        return [((g.strip().strip("*").upper()[:1] == t) if g.strip() else False) for g, t in zip(gens, truths)]

    rng = np.random.default_rng(11)
    rows = []
    for i in range(args.n):
        pool = [w for w in words if (w in ANIMALS) == (i % 2 == 0)]
        x = pool[int(rng.integers(len(pool)))]
        ys = [w for w in words if (w in ANIMALS) != (x in ANIMALS) and (w in RIDDEN) != (x in RIDDEN) and w[0] != x[0]]
        zs = [w for w in words if w != x and (w in ANIMALS) == (x in ANIMALS) and (w in RIDDEN) == (x in RIDDEN) and w[0] != x[0]]
        zs = zs or [w for w in words if w != x and (w in ANIMALS) == (x in ANIMALS) and w[0] != x[0]]
        assert ys and zs, x
        y = ys[int(rng.integers(len(ys)))]
        z = zs[int(rng.integers(len(zs)))]
        q = int(rng.integers(1, stimuli.N_SLOTS + 1))
        rows.append(dict(scene=i, X=x, Y=y, Z=z, q=q, tile=bank[x][i % len(bank[x])]))
    S = pd.DataFrame(rows)
    S["calibration"] = S.scene < args.n_cal
    own_img = [stimuli.scene(r.q, r.tile) for r in S.itertuples()]
    prt_img = [stimuli.scene(r.q, stimuli.word_tile(r.X)) for r in S.itertuples()]
    blank = stimuli.scene(None, None)
    n_img = stimuli.N_SLOTS * 49
    slot_idx = {s: stimuli.slot_tokens(stimuli.COLS, stimuli.ROWS, s) for s in range(1, stimuli.N_SLOTS + 1)}
    slot_pos = lambda row, i: [[j for j, t in enumerate(row) if t == image_id][k] for k in slot_idx[int(S.q.iloc[i])]]  # noqa: E731
    dec_pos = lambda row, i: [len(row) - 1]  # noqa: E731
    all_pos = hooks.image_positions(image_id)
    stimuli.contact_sheet(
        [(f"photo: {r.X} in slot {r.q}", own_img[i]) for i, r in enumerate(S.head(2).itertuples())] + [(f"printed: {r.X} in slot {r.q}", prt_img[i]) for i, r in enumerate(S.head(2).itertuples())],
        4,
        run.path("stimuli.png"),
    )

    # the clean image-token residuals at the read layers (the image precedes every question, so one capture serves all four)
    cap_txt = [lm.chat_media([INTRO, "IMAGE", TASKS["say"].format(q=r.q)]) for r in S.itertuples()]
    _, R = lm.forward(cap_txt, own_img, layers, where=lambda ids: [j for j, t in enumerate(ids) if t == image_id])
    assert R[layers[0]].shape[1] == n_img, R[layers[0]].shape
    H = {L: R[L].half().numpy() for L in layers}
    del R
    MU = {L: H[L].reshape(-1, d_model).astype(np.float32).mean(0) for L in layers}
    ids_x, ids_y, ids_z = [[sid[w][0] for w in S[c]] for c in ("X", "Y", "Z")]
    ids_animal = lm.single_ids(["animal", " animal", "Animal", " Animal"])
    assert ids_animal, "no single-token form of 'animal'"
    U = {}
    U["own name"] = {L: lens.word_directions(lm, L, ids_x, "J") for L in layers}
    U["other name"] = {L: lens.word_directions(lm, L, ids_y, "J") for L in layers}
    U["random direction"] = {L: lens.unit_rows(len(S), d_model, 200 + L) for L in layers}
    U["own name, W_U only"] = {L: lens.word_directions(lm, L, ids_x, "W_U") for L in layers}
    if lm.jpp is not None:
        U["own name, J++"] = {L: lens.word_directions(lm, L, ids_x, "J++") for L in layers}
    U["the word 'animal'"] = {L: lens.word_directions(lm, L, [ids_animal[0]] * len(S), "J") for L in layers}
    U["own name, orthogonal to 'animal'"] = {}
    for L in layers:
        a, c = U["own name"][L][:, :, 0], U["the word 'animal'"][L][:, :, 0]
        v = a - (a * c).sum(1, keepdims=True) * c
        U["own name, orthogonal to 'animal'"][L] = (v / np.linalg.norm(v, axis=1, keepdims=True))[:, :, None]
    U["top principal direction of the slot"] = {}
    for L in layers:
        V = []
        for i, r in enumerate(S.itertuples()):
            Xs = torch.as_tensor(H[L][i, slot_idx[r.q]], dtype=torch.float32, device=lm.device)
            Xs = Xs - Xs.mean(0, keepdim=True)
            V.append(torch.linalg.svd(Xs, full_matrices=False)[2][0].cpu().numpy())
        U["top principal direction of the slot"][L] = np.stack(V)[:, :, None]
    U["lens part, 256 directions"] = {L: lens.lens_subspace(lm, L, 256) for L in layers}
    U["random part, 256 directions"] = {L: lens.random_subspace(d_model, 256, 100 + L) for L in layers}
    REMOVALS = list(U)
    U["same-kind name"] = {L: lens.word_directions(lm, L, ids_z, "J") for L in layers}
    U["other name, W_U only"] = {L: lens.word_directions(lm, L, ids_y, "W_U") for L in layers}
    U["random direction, blank"] = {L: lens.unit_rows(len(S), d_model, 300 + L) for L in layers}
    # sigma_L: the mean size of a slot token's own-name component, the unit of the steering dose
    sigma = {L: float(np.mean([np.abs(H[L][i, slot_idx[r.q]].astype(np.float32) @ U["own name"][L][i, :, 0]).mean() for i, r in enumerate(S.itertuples())])) for L in layers}
    energy = {}  # share of the centred slot tokens' energy each removed part holds
    for name in REMOVALS:
        e = []
        for L in layers:
            Qn = np.asarray(U[name][L])
            for i, r in enumerate(S.itertuples()):
                Xs = H[L][i, slot_idx[r.q]].astype(np.float32) - MU[L]
                e.append(float(((Xs @ (Qn[i] if Qn.ndim == 3 else Qn)) ** 2).sum() / (Xs**2).sum()))
        energy[name] = float(np.mean(e))
    print("sigma_L: " + ", ".join(f"L{L} {sigma[L]:.1f}" for L in layers[:: max(1, len(layers) // 4)]))
    print("energy share removed: " + ", ".join(f"{k} {v:.3f}" for k, v in energy.items()) + f"  [{time.time() - t0:.0f}s]")

    # ---- the arms
    res = []
    TX = {task: [truth(task, r.X) for r in S.itertuples()] for task in TASKS}
    TY = {task: [truth(task, r.Y) for r in S.itertuples()] for task in TASKS}
    TZ = {task: [truth(task, r.Z) for r in S.itertuples()] for task in TASKS}

    def arm_run(block, arm, task, images, hk, follow, alpha=0.0):
        """one arm of one question: 'correct' scores against X, 'follow' against the steering word (Y or Z) when given"""
        tx = [lm.chat_media([INTRO, "IMAGE", TASKS[task].format(q=r.q)]) for r in S.itertuples()]
        gen = lm.generate(tx, images, 5, hooks=hk)
        ok, fo = score(task, gen, TX[task]), (score(task, gen, follow) if follow is not None else [None] * len(S))
        for i, r in enumerate(S.itertuples()):
            res.append(dict(block=block, arm=arm, task=task, scene=r.scene, X=r.X, Y=r.Y, Z=r.Z, q=r.q, alpha=alpha, truth=TX[task][i], generation=gen[i], correct=ok[i], follow=fo[i]))
        return float(np.mean(ok)), (float(np.mean(fo)) if follow is not None else float("nan"))

    ablate = {name: hooks.ablate_directions(U[name], slot_pos, layers, mu=MU) for name in REMOVALS}
    for arm in ["clean"] + REMOVALS:
        acc = {task: arm_run("own picture", arm, task, own_img, None if arm == "clean" else ablate[arm], TY[task]) for task in TASKS}
        print(f"own picture   {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in TASKS) + f"  [{time.time() - t0:.0f}s]")
    for arm, hk in [("clean", None), ("own name", ablate["own name"])]:
        acc = {task: arm_run("printed word", arm, task, prt_img, hk, None) for task in TASKS}
        print(f"printed word  {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in TASKS) + f"  [{time.time() - t0:.0f}s]")

    def first_ids(task, t):
        return sid[t] if task == "say" else lm.single_ids([t, " " + t])

    def logp(task, images, hk):
        """log p of the correct first token (summed over its single-token forms) at the decision position, per scene"""
        tx = [lm.chat_media([INTRO, "IMAGE", TASKS[task].format(q=r.q)]) for r in S.itertuples()]
        lg, _ = lm.forward(tx, images, hooks=hk)
        lp = torch.log_softmax(lg.float(), -1)
        return np.array([float(torch.logsumexp(lp[i, first_ids(task, TX[task][i])], 0)) for i in range(len(S))])

    LP = {}
    for block, images in (("own picture", own_img), ("printed word", prt_img)):
        for arm in ["clean"] + REMOVALS if block == "own picture" else ["clean", "own name"]:
            for task in TASKS:
                LP[(block, arm, task)] = logp(task, images, None if arm == "clean" else ablate[arm])
        print(f"{block}: log p captured for {len(LP)} arm-task pairs  [{time.time() - t0:.0f}s]")
    run.save_csv(pd.DataFrame([dict(block=b, arm=a, task=t, scene=int(S.scene.iloc[i]), logp=float(v[i])) for (b, a, t), v in LP.items() for i in range(len(S))]), "logp.csv")
    SITES = {"image tokens": slot_pos, "all image tokens": all_pos, "decision position": dec_pos}
    STEER = [("other name", TY), ("random direction", TY), ("same-kind name", TZ), ("other name, W_U only", TY)]
    for site, pos_of in SITES.items():
        for arm, follow in STEER:
            for a in ALPHAS:
                hk = hooks.add_directions(U[arm], pos_of, layers, {L: a * sigma[L] for L in layers})
                acc = {task: arm_run(f"inject at the {site}", arm, task, own_img, hk, follow[task], alpha=a) for task in TASKS}
                print(
                    f"{site:17s} x{a:.0f} {arm:24s} follow "
                    + "  ".join(f"{t} {acc[t][1]:.2f}" for t in TASKS)
                    + " | keep "
                    + "  ".join(f"{t} {acc[t][0]:.2f}" for t in TASKS)
                    + f"  [{time.time() - t0:.0f}s]"
                )
    blank_imgs = [blank] * len(S)
    for arm, a in [("none", 0.0)] + [("own name", a) for a in ALPHAS] + [("random direction, blank", 8.0)]:
        hk = None if a == 0 else hooks.add_directions(U[arm], slot_pos, layers, {L: a * sigma[L] for L in layers})
        acc = {task: arm_run("blank canvas", arm, task, blank_imgs, hk, None, alpha=a) for task in TASKS}
        print(f"blank X x{a:.0f}   {arm:36s} " + "  ".join(f"{t} {acc[t][0]:.2f}" for t in TASKS) + f"  [{time.time() - t0:.0f}s]")
    G1 = pd.DataFrame(res)
    run.save_csv(G1, "name_steering.csv")

    # ---- the rules (RECORD.md 03:55 / 04:00)
    HELD = S.scene[~S.calibration].values if args.n > args.n_cal else S.scene.values

    def vals(block, arm, task, alpha=0.0, col="correct", scenes=None):
        d = G1[(G1.block == block) & (G1.arm == arm) & (G1.task == task) & (G1.alpha == alpha)]
        v = d.set_index("scene")[col].astype(float).reindex(S.scene).fillna(0).values
        return v if scenes is None else v[np.isin(S.scene.values, scenes)]

    def boot(v, scenes=None):
        return boot_scene(v, S.scene.values if scenes is None else S.scene.values[np.isin(S.scene.values, scenes)])

    out = dict(n=int(len(S)), n_held=int(len(HELD)), sigma=sigma, energy=energy, accuracy={}, follow={}, D={}, verdict={})
    for (block, arm, a), _ in G1.groupby(["block", "arm", "alpha"]):
        key = f"{block} | {arm} | {a:g}"
        out["accuracy"][key] = {task: boot(vals(block, arm, task, a)) for task in TASKS}
        if block.startswith("inject") or block == "own picture":
            out["follow"][key] = {task: boot(vals(block, arm, task, a, "follow")) for task in TASKS}
    clean = {task: out["accuracy"]["own picture | clean | 0"][task][0] for task in TASKS}
    valid = {task: clean[task] >= (0.9 if task == "say" else 0.85) for task in TASKS}
    rules_ok = [t for t in RULES if valid[t]]
    D = {task: boot(vals("own picture", "other name", task) - vals("own picture", "own name", task)) for task in TASKS}
    out["D"] = D

    def r1(task):
        m, lo, hi = D[task]
        return "necessary" if m >= 0.25 and lo > 0 else "not needed" if m <= 0.10 and lo > -0.15 and hi < 0.15 else "undetermined"

    out["verdict"]["R0"] = {task: bool(valid[task]) for task in TASKS}
    out["verdict"]["R1"] = {task: r1(task) for task in TASKS}
    out["dlogp"] = {f"{b} | {a}": {t: boot(LP[(b, a, t)] - LP[(b, "clean", t)]) for t in TASKS} for (b, a, _) in LP if a != "clean"}
    own, other, cl = ("own picture", "own name"), ("own picture", "other name"), ("own picture", "clean")
    DN = {task: boot((LP[(*own, task)] - LP[(*cl, task)]) - (LP[(*other, task)] - LP[(*cl, task)])) for task in TASKS}
    out["D_nats"] = DN

    def r1n(task):
        m, lo, hi = DN[task]
        return "necessary" if m <= -0.5 and hi < 0 else "not needed" if m >= -0.2 and lo > -0.3 and hi < 0.3 else "undetermined"

    out["verdict"]["R1_nats"] = {task: r1n(task) for task in TASKS}
    gn = boot((LP[(*own, "say")] - LP[(*other, "say")]) - (np.mean([LP[(*own, t)] - LP[(*other, t)] for t in rules_ok], 0) if rules_ok else 0))
    out["G_nats"] = gn
    r2n = (
        "positive control failed"
        if not valid["say"] or not rules_ok
        else "use runs through the handle"
        if all(r1n(t) == "necessary" for t in ["say"] + rules_ok) and gn[1] > -0.3 and gn[2] < 0.3
        else "the handle is for saying only"
        if r1n("say") == "necessary" and all(r1n(t) == "not needed" for t in rules_ok)
        else "saying needs it more"
        if gn[0] <= -0.3 and gn[2] < 0
        else "undetermined at this n"
    )
    out["verdict"]["R2_nats"] = r2n
    g_vals = (
        vals("own picture", "other name", "say")
        - vals("own picture", "own name", "say")
        - (np.mean([vals("own picture", "other name", t) - vals("own picture", "own name", t) for t in rules_ok], 0) if rules_ok else 0)
    )
    G = boot(g_vals)
    out["G"] = G
    if not valid["say"] or not rules_ok:
        r2 = "positive control failed"
    elif all(r1(t) == "necessary" for t in ["say"] + rules_ok) and G[1] > -0.15 and G[2] < 0.15:
        r2 = "use runs through the handle"
    elif r1("say") == "necessary" and all(r1(t) == "not needed" for t in rules_ok):
        r2 = "the handle is for saying only"
    elif G[0] >= 0.15 and G[1] > 0:
        r2 = "saying needs it more"
    else:
        r2 = "undetermined at this n"
    out["verdict"]["R2"] = r2
    ctrl = {arm: max(abs(clean[t] - out["accuracy"][f"own picture | {arm} | 0"][t][0]) for t in TASKS) for arm in ["random direction", "own name, W_U only", "other name"]}
    own_drop = {t: clean[t] - out["accuracy"]["own picture | own name | 0"][t][0] for t in TASKS}
    pc_drop = {t: clean[t] - out["accuracy"]["own picture | top principal direction of the slot | 0"][t][0] for t in TASKS}
    out["verdict"]["R3"] = dict(
        controls_within_0_10=bool(max(ctrl.values()) <= 0.10),
        control_max_shift=ctrl,
        energy_explains=bool(any(pc_drop[t] >= own_drop[t] - 0.10 and own_drop[t] >= 0.25 for t in TASKS)),
    )

    # R4-R6: the steering sites. alpha* per site = the smallest alpha at which say follows Y on >= 0.5 of the calibration scenes;
    # verdicts on the held-out scenes
    CAL = S.scene[S.calibration].values if args.n > args.n_cal else S.scene.values
    site_out = {}
    for site in SITES:
        blk = f"inject at the {site}"
        a_star = next((a for a in ALPHAS if vals(blk, "other name", "say", a, "follow", CAL).mean() >= 0.5), None)
        o = dict(alpha_star=a_star)
        if a_star is not None:
            rulesY = np.mean([vals(blk, "other name", t, a_star, "follow", HELD) for t in RULES], 0)
            rulesR = np.mean([vals(blk, "random direction", t, a_star, "follow", HELD) for t in RULES], 0)
            o["say_follows_Y"] = boot(vals(blk, "other name", "say", a_star, "follow", HELD), HELD)
            o["rules_follow_Y"] = boot(rulesY, HELD)
            o["rules_follow_random"] = boot(rulesR, HELD)
            o["rules_Y_minus_random"] = boot(rulesY - rulesR, HELD)
            o["Z_say_follows_Z"] = boot(vals(blk, "same-kind name", "say", a_star, "follow", HELD), HELD)
            o["Z_rules_flip"] = boot(1 - np.mean([vals(blk, "same-kind name", t, a_star, "correct", HELD) for t in RULES], 0), HELD)
            o["WU_say_follows_Y"] = boot(vals(blk, "other name, W_U only", "say", a_star, "follow", HELD), HELD)
            o["random_say_keeps_X"] = boot(vals(blk, "random direction", "say", a_star, "correct", HELD), HELD)
            o["letter_follows_Y"] = boot(vals(blk, "other name", "letter", a_star, "follow", HELD), HELD)
            clean_keep = vals("own picture", "clean", "say", 0, "correct", HELD).mean()
            o["collateral_ok"] = bool(o["random_say_keeps_X"][0] >= clean_keep - 0.10)
            o["J_specific"] = bool(o["WU_say_follows_Y"][0] <= 0.10)
            d = o["rules_Y_minus_random"]
            o["verdict"] = (
                "uninterpretable (the norm alone breaks the picture)"
                if not o["collateral_ok"]
                else "seen"
                if d[0] >= 0.30 and d[1] > 0 and o["Z_rules_flip"][0] <= 0.10
                else "only said"
                if o["rules_follow_Y"][0] <= o["rules_follow_random"][0] + 0.10
                else "undetermined"
            )
            o["rules_follow_Y_vals"] = rulesY.tolist()
        else:
            o["verdict"] = "naming never follows the name at any dose"
        site_out[site] = o
    out["dissociation"], r6 = {}, {}
    for isite in ["image tokens", "all image tokens"]:
        if site_out[isite].get("alpha_star") is not None and site_out["decision position"].get("alpha_star") is not None:
            dis = np.asarray(site_out[isite]["rules_follow_Y_vals"]) - np.asarray(site_out["decision position"]["rules_follow_Y_vals"])
            out["dissociation"][isite] = boot(dis, HELD)
            r6[isite] = (
                "the same direction, written on the picture, is seen; written at the answer, it is only said"
                if out["dissociation"][isite][0] >= 0.30 and out["dissociation"][isite][1] > 0
                else "no site dissociation at this n"
            )
    r6_txt = "; ".join(f"{s_}: {t_} ({fmt(out['dissociation'][s_])})" for s_, t_ in r6.items()) or "not testable (no alpha* at one site)"
    out["verdict"]["R6"] = r6_txt
    for s in SITES:
        site_out[s].pop("rules_follow_Y_vals", None)
    out["sites"] = site_out
    out["verdict"]["R4"] = {s: site_out[s]["verdict"] for s in SITES}
    out["verdict"]["R5"] = {
        task: next(
            (a for a in ALPHAS if out["accuracy"][f"blank canvas | own name | {a:g}"][task][0] >= 0.5 and out["accuracy"]["blank canvas | random direction, blank | 8"][task][0] <= 0.1),
            None,
        )
        for task in TASKS
    }
    print("D (other name removed minus own name removed): " + "  ".join(f"{t} {fmt(D[t])} {r1(t)}" for t in TASKS))
    print("D in nats: " + "  ".join(f"{t} {fmt(DN[t])} {r1n(t)}" for t in TASKS) + f" | G {fmt(gn)} -> R2 (nats): {r2n}")
    print(f"G {fmt(G)} -> R2: {r2}; R3 {out['verdict']['R3']}")
    for s in SITES:
        o = site_out[s]
        print(f"{s}: alpha* {o['alpha_star']} -> {o['verdict']}" + ("" if o["alpha_star"] is None else f"; say follows Y {fmt(o['say_follows_Y'])}, rules follow Y {fmt(o['rules_follow_Y'])}"))
    print(f"R6: {r6_txt}; R5 blank first alpha {out['verdict']['R5']}")
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
