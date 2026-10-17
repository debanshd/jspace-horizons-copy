"""Printed word: one word's lens direction decides whether the model reads a printed label or sees the object (A1 in the notes).

A COCO crop of X with the word Y printed on a white band across its lower part, alone on a one-slot canvas. Question: "What object
is this?" Interventions on every image token at the read layers: directional ablation (rank 1) of Y's lens direction (the label),
X's (the object), another word Z's, a random direction, and mean-ablation of Y's and X's; and the 256-dimensional lens subspace
removed only on the label band's tokens or only on the object's rows, each beside a random 256-dimensional part. Answers are judged
by the LLM judge (says X / says Y). Also on the clean input: the lens readout of X and Y, "Is there a X in the picture?", and,
after the model's own answer, "Did your answer come from the object shown in the picture or from the printed word?".
Perception gates: the crop alone is named X, the word alone is read as Y.

Outputs in runs/printed_word/<model>/: printed_word.csv (arm x scene: answer, says_X, says_Y), scenes.csv (per scene: the
self-report, the source report, lens ranks), summary.json, stimuli.png (the first six inputs), crops.json. Examples 7 and 8.
"""

import time

import numpy as np
import pandas as pd
from PIL import Image

from jspace import hooks, lens, stimuli
from jspace.analysis import boot_scene, fmt, rate
from jspace.config import RESULTS_DIR, require_data
from jspace.judge import Judge
from jspace.model import Session
from jspace.runs import Run, parser

QUESTION = "What object is this? Reply with the noun only."
SOURCE_QUESTION = "Did your answer come from the object shown in the picture or from the printed word? Reply with one word: object or word."


def main():
    ap = parser(__doc__, "printed_word")
    ap.add_argument("--n", type=int, default=64, help="scenes")
    ap.add_argument("--crop-set", type=int, default=0)
    ap.add_argument("--fresh-judge", action="store_true", help="ignore the verdicts recorded in results/judge_cache.json")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "printed_word")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank, sid = crops.words, crops.tiles, crops.ids
    judge = Judge(run.dir, [] if args.fresh_judge else [RESULTS_DIR / "judge_cache.json"])
    at_image = hooks.image_positions(lm.image_token_id)
    t0 = time.time()

    def chat2(answer, q2):
        """a second user turn after the model's own answer"""
        return lm.chat_messages(
            [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                {"role": "user", "content": [{"type": "text", "text": q2}]},
            ]
        )

    def first_of(gen, options):
        g = gen.strip().lower()
        return next((o for o in options if g.startswith(o)), None)

    rng = np.random.default_rng(1)
    rows = []
    for i in range(args.n):
        x, y, z = [words[j] for j in rng.choice(len(words), 3, replace=False)]
        rows.append(dict(scene=i, X=x, Y=y, Z=z, crop=bank[x][i % len(bank[x])]))
    S = pd.DataFrame(rows)

    def canv(tiles):
        return [stimuli.canvas(1, 1, {1: t}) for t in tiles]

    img_lab = canv([stimuli.label_tile(r.crop, r.Y) for r in S.itertuples()])
    img_obj = canv([r.crop for r in S.itertuples()])
    img_word = canv([stimuli.label_tile(Image.new("RGB", (stimuli.CROP, stimuli.CROP), stimuli.BG), r.Y) for r in S.itertuples()])
    stimuli.contact_sheet([(f"{r.X} + '{r.Y}'", img_lab[i]) for i, r in enumerate(S.head(6).itertuples())], 3, run.path("stimuli.png"))
    texts = [lm.chat_media(["IMAGE", QUESTION])] * len(S)
    gate_obj = judge.name_match(lm.generate(texts, img_obj, 6), S.X.tolist())
    gate_word = judge.name_match(lm.generate(texts, img_word, 6), S.Y.tolist())
    print(f"gates: the crop alone named X {rate(gate_obj):.2f}; the word alone read as Y {rate(gate_word):.2f}  [{time.time() - t0:.0f}s]")

    n_img = 49
    tid = lambda ws: [int(sid[w][0]) for w in ws]  # noqa: E731
    Qx = {L: lens.word_directions(lm, L, tid(S.X)) for L in lm.layers}
    Qy = {L: lens.word_directions(lm, L, tid(S.Y)) for L in lm.layers}
    Qz = {L: lens.word_directions(lm, L, tid(S.Z)) for L in lm.layers}
    Qr = {L: np.repeat(lens.orth(np.random.default_rng(7 + L).normal(0, 1, (1, Qx[L].shape[1])))[None], len(S), 0) for L in lm.layers}
    _, R = lm.forward(texts, img_lab, lm.layers, where=lens.where_image(lm, n_img))
    MU = {L: R[L][:, :-1].reshape(-1, R[L].shape[-1]).mean(0).numpy() for L in lm.layers}
    del R
    # clean readout: X and Y on the image tokens and at the decision position (J-lens top 10, best read layer)
    rj, rl, _, _ = lens.ranks_at(lm, texts, img_lab, lens.where_image(lm, n_img), n_img, [sid[w] for w in words], 16)
    wi = {w: j for j, w in enumerate(words)}
    RO = pd.DataFrame(
        [
            dict(
                scene=r.scene,
                X_img=int(rj[i, :, :-1, wi[r.X]].min()),
                Y_img=int(rj[i, :, :-1, wi[r.Y]].min()),
                X_dec=int(rj[i, :, -1, wi[r.X]].min()),
                Y_dec=int(rj[i, :, -1, wi[r.Y]].min()),
                X_img_logit=int(rl[i, :, :-1, wi[r.X]].min()),
                Y_img_logit=int(rl[i, :, :-1, wi[r.Y]].min()),
            )
            for i, r in enumerate(S.itertuples())
        ]
    ).set_index("scene")

    def ablate(Q, mu=None):
        return hooks.ablate_directions(Q, at_image, lm.layers, mu=mu)

    arms = [
        ("clean", None),
        ("ablate label word Y", ablate(Qy)),
        ("ablate object word X", ablate(Qx)),
        ("ablate other word Z", ablate(Qz)),
        ("ablate random direction", ablate(Qr)),
        ("mean-ablate label word Y", ablate(Qy, MU)),
        ("mean-ablate object word X", ablate(Qx, MU)),
    ]
    # source-localized arms: the 256-d lens subspace removed only on the label band's tokens (rows 5-6 of the 7 x 7 grid) or only
    # on the object's rows (0-3), each beside a random part of the same rank on the same tokens
    Qs = {L: lens.lens_subspace(lm, L, 256) for L in lm.layers}
    Qrand = {L: lens.random_subspace(Qs[L].shape[0], 256, 100 + L) for L in lm.layers}

    def band(grid_rows):
        return lambda row, i: [[j for j, t in enumerate(row) if t == lm.image_token_id][k] for k in range(n_img) if k // 7 in grid_rows]

    arms += [
        ("ablate J part on the label band", hooks.ablate_subspace(Qs, band((5, 6)), lm.layers)),
        ("ablate random part on the label band", hooks.ablate_subspace(Qrand, band((5, 6)), lm.layers)),
        ("ablate J part on the object rows", hooks.ablate_subspace(Qs, band((0, 1, 2, 3)), lm.layers)),
        ("ablate random part on the object rows", hooks.ablate_subspace(Qrand, band((0, 1, 2, 3)), lm.layers)),
    ]
    res, GEN = [], {}
    for name, hk in arms:
        gen = lm.generate(texts, img_lab, 6, hooks=hk)
        GEN[name] = gen
        sx, sy = judge.name_match(gen, S.X.tolist()), judge.name_match(gen, S.Y.tolist())
        for i, r in enumerate(S.itertuples()):
            res.append(dict(arm=name, scene=r.scene, X=r.X, Y=r.Y, Z=r.Z, generation=gen[i], says_X=sx[i], says_Y=sy[i]))
        print(f"{name:26s} says X {rate(sx):.2f}  says Y {rate(sy):.2f}  [{time.time() - t0:.0f}s]")
    A1 = pd.DataFrame(res)
    # self-report and source report on the clean input
    ask = [lm.chat_media(["IMAGE", f"Is there a {r.X} in the picture? Answer yes or no."]) for r in S.itertuples()]
    yes_X = [first_of(g, ("yes", "no")) for g in lm.generate(ask, img_lab, 4)]
    src = [first_of(g, ("object", "word")) for g in lm.generate([chat2(GEN["clean"][i], SOURCE_QUESTION) for i in range(len(S))], img_lab, 4)]
    clean = A1[A1.arm == "clean"].set_index("scene")
    S["says_yes_X"] = [v == "yes" if v else None for v in yes_X]
    S["source_report"] = src
    S["clean_says_X"], S["clean_says_Y"] = clean.says_X.values, clean.says_Y.values
    run.save_csv(S.drop(columns=["crop"]).merge(RO, left_on="scene", right_index=True), "scenes.csv")
    run.save_csv(A1, "printed_word.csv")
    judge.label_sample(A1[A1.arm.isin(["clean", "ablate label word Y"])], "printed_word", label_col="says_X")

    # ---- summary
    def pb(arm, col):
        return A1[A1.arm == arm].set_index("scene")[col].astype(float).fillna(0)

    res_ = dict(n=int(len(S)), gates=dict(crop_named_X=rate(gate_obj), word_read_as_Y=rate(gate_word)), arms={}, readout={}, self_report={}, source_report={})
    for name, _ in arms:
        res_["arms"][name] = dict(says_X=boot_scene(pb(name, "says_X").values, S.scene.values), says_Y=boot_scene(pb(name, "says_Y").values, S.scene.values))
    label_following = res_["arms"]["clean"]["says_Y"][0]

    def paired(a):
        return boot_scene((pb(a, "says_X") - pb("clean", "says_X")).values, S.scene.values)

    res_["rescue_Y"] = paired("ablate label word Y")
    res_["rescue_Y_mean"] = paired("mean-ablate label word Y")
    res_["loss_X"] = boot_scene((pb("ablate object word X", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
    res_["ctrl_Z"], res_["ctrl_random"] = paired("ablate other word Z"), paired("ablate random direction")
    res_["rescue_band"], res_["ctrl_band_random"] = paired("ablate J part on the label band"), paired("ablate random part on the label band")
    res_["loss_object_rows"] = boot_scene((pb("ablate J part on the object rows", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
    res_["ctrl_object_rows_random"] = boot_scene((pb("ablate random part on the object rows", "says_Y") - pb("clean", "says_Y")).values, S.scene.values)
    for col, lab in (
        ("X_img", "X on image tokens"),
        ("Y_img", "Y on image tokens"),
        ("X_dec", "X at the decision position"),
        ("Y_dec", "Y at the decision position"),
        ("X_img_logit", "X on image tokens (logit lens)"),
    ):
        res_["readout"][lab] = boot_scene(lens.readable(RO[col].values).astype(float), S.scene.values)
    lf = S[S.clean_says_Y == True]  # noqa: E712 - verdicts can be None
    on = S[S.clean_says_X == True]  # noqa: E712
    res_["self_report"] = dict(says_yes_X_all=rate(S.says_yes_X.tolist()), says_yes_X_when_following_label=rate(lf.says_yes_X.tolist()), n_following=int(len(lf)))
    res_["source_report"] = dict(
        word_when_following_label=rate([v == "word" if v else None for v in lf.source_report]),
        word_when_naming_object=rate([v == "word" if v else None for v in on.source_report]),
        n_following=int(len(lf)),
        n_naming=int(len(on)),
    )
    # the rules, frozen before the run (RECORD.md 03:40): label-following >= 0.3 or stop; rescue when says-X rises >= 0.3 under the
    # Y ablation (interval > 0) with the other-word and random arms within +-0.05
    gate_ok = res_["gates"]["crop_named_X"] >= 0.9 and res_["gates"]["word_read_as_Y"] >= 0.9
    rz, rr, ry = res_["ctrl_Z"], res_["ctrl_random"], res_["rescue_Y"]
    rb, rbr = res_["rescue_band"], res_["ctrl_band_random"]
    res_["verdict"] = (
        "perception gate failed"
        if not gate_ok
        else "premise failed: the label is followed on fewer than 0.3 of items"
        if label_following < 0.3
        else "rescue: removing the label word's one direction returns the model to the object"
        if ry[0] >= 0.3 and ry[1] > 0 and abs(rz[0]) <= 0.05 and abs(rr[0]) <= 0.05
        else "rescue: removing the lens part on the label's own tokens returns the model to the object"
        if rb[0] >= 0.3 and rb[1] > 0 and abs(rbr[0]) <= 0.1
        else "no rescue"
        if ry[2] < 0.3 and rb[2] < 0.3
        else "undetermined at this n"
    )
    res_["seconds"] = time.time() - t0
    run.save_json(res_, "summary.json")
    print("readout (clean, top-10): " + ", ".join(f"{k} {v[0]:.2f}" for k, v in res_["readout"].items()))
    print(f"label-following {label_following:.2f}; rescue (Y direction) {fmt(ry)}, mean-ablated {fmt(res_['rescue_Y_mean'])}; controls Z {fmt(rz)} random {fmt(rr)}")
    print(f"source report says 'word' on {res_['source_report']['word_when_following_label']:.2f} of label-following and {res_['source_report']['word_when_naming_object']:.2f} of object-naming items")
    print(f"-> {res_['verdict']}")


if __name__ == "__main__":
    main()
