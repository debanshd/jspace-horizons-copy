"""GQA ablation: is what the J-lens reads off a natural photo's image tokens necessary for answering questions about it? (N1)

GQA testdev-balanced photos with the benchmark's own open "query" questions whose answer is one token, one question per image in
each of three bins of program length (2, 3, 4 or more steps). 48 images, disjoint from the 16 of the pilot. Arms, all at the read
layers on every image token: clean; no image (the language prior); ablate the lens subspace (rank 256 and 64: the top right
singular directions of W_U J_L); an equal-rank random subspace; the top principal components of the image-token states (the
energy-matched control); the top directions of W_U alone; J++ where fitted; keep only the lens subspace; the answer word's own
lens direction (rank 1) against another question's answer word; and mean-ablation of the 256-dimensional parts. Outcomes per
question: log p of the reference answer's first token, and the greedy answer judged by the LLM judge (a regex score beside it).

Outputs in runs/gqa_ablation/<model>/: gqa_ablation.csv (arm x question: answer, judged correct, log p, lens ranks of the
reference word on the image tokens and at the answer position), summary.json, stimuli.png. Example 6.
"""

import time

import numpy as np
import pandas as pd
import torch

from jspace import hooks, lens, stimuli
from jspace.analysis import boot_scene, fmt, noun_matches
from jspace.config import RESULTS_DIR, require_data
from jspace.judge import Judge
from jspace.model import Session
from jspace.runs import Run, parser

BINS = {"2": [2], "3": [3], "4+": [4, 5, 6, 7, 8, 9]}
# the 16 images of the pilot run, excluded so the confirmatory sample is disjoint from it
PILOT_IMAGES = {"n24526", "n249639", "n25275", "n282436", "n302387", "n315887", "n346247", "n350766", "n413319", "n437192", "n450919", "n523165", "n556604", "n560243", "n59676", "n67005"}


def main():
    ap = parser(__doc__, "gqa_ablation")
    ap.add_argument("--n-images", type=int, default=48)
    ap.add_argument("--include-pilot", action="store_true", help="do not exclude the pilot's 16 images")
    ap.add_argument("--ranks", default="256,64", help="subspace ranks (256 is always needed)")
    ap.add_argument("--no-mean-ablation", action="store_true", help="skip the mean-ablation arms")
    ap.add_argument("--fresh-judge", action="store_true", help="ignore the verdicts recorded in results/judge_cache.json")
    args = ap.parse_args()
    require_data(args.data_dir, "gqa")
    ranks = [int(x) for x in args.ranks.split(",")]
    assert 256 in ranks
    run = Run(args, "gqa_ablation")
    lm = Session(args.model)
    run.write_config(lm)
    judge = Judge(run.dir, [] if args.fresh_judge else [RESULTS_DIR / "judge_cache.json"])
    layers, image_id = lm.layers, lm.image_token_id
    at_image = hooks.image_positions(image_id)
    gqa = args.data_dir / "gqa"
    t0 = time.time()

    def where_image(n_image):
        """a row's last n_image image tokens, then its last token"""
        return lambda ids: [j for j, t in enumerate(ids) if t == image_id][-n_image:] + [len(ids) - 1]

    def pca_subspace(H, r):
        """top-r principal directions of the image-token states H [n, d] (centred), orthonormal [d, r]"""
        X = torch.as_tensor(H, dtype=torch.float32, device=lm.device)
        X = X - X.mean(0, keepdim=True)
        q = min(r + 8, X.shape[1] - 1, X.shape[0] - 1)
        if q < r + 8:
            print(f"PCA control: only {X.shape[0]} samples, rank capped at {q - 8}")
        U, S_, V = torch.pca_lowrank(X, q=q, center=False)  # randomized: draws from the generator Session seeded
        return lens.orth(V[:, :r].T.cpu().numpy())

    gq = pd.read_parquet(gqa / "testdev_balanced_instructions" / "testdev-00000-of-00001.parquet")
    gq["plen"] = gq.semantic.apply(len)
    gq["structural"] = gq.types.apply(lambda t: t["structural"])
    gq = gq[(gq.structural == "query") & (gq.answer.str.split().str.len() == 1)].copy()
    gq["ans_ids"] = gq.answer.apply(lambda a: lm.single_ids([a, " " + a, a.capitalize(), " " + a.capitalize()]))
    gq = gq[gq.ans_ids.apply(len) > 0]
    gq["bin"] = gq.plen.apply(lambda n: next((b for b, v in BINS.items() if n in v), None))
    gq = gq[gq.bin.notna()]
    rng = np.random.default_rng(0)
    imgs_ok = [i for i, g in gq.groupby("imageId") if g.bin.nunique() == 3]
    if not args.include_pilot:
        imgs_ok = [i for i in imgs_ok if str(i) not in PILOT_IMAGES]
        print(f"excluding {len(PILOT_IMAGES)} pilot images; {len(imgs_ok)} eligible images remain")
    pick = [imgs_ok[j] for j in rng.permutation(len(imgs_ok))[: args.n_images]]
    rows_q = []
    for iid in pick:
        g = gq[gq.imageId == iid]
        for b in BINS:
            gb = g[g.bin == b]
            rows_q.append(gb.iloc[int(rng.integers(len(gb)))])
    GQ = pd.DataFrame(rows_q).reset_index(drop=True)
    imtab = pd.read_parquet(gqa / "testdev_balanced_images" / "testdev-00000-of-00001.parquet")
    imtab = imtab[imtab.id.isin(pick)].set_index("id")
    IM = {iid: stimuli.fit_photo(imtab.loc[iid, "image"]["bytes"]) for iid in pick}
    print(f"GQA: {len(pick)} images x 3 questions (program length bins {list(BINS)}), {len(gq)} eligible query questions in the pool  [{time.time() - t0:.0f}s]")
    stimuli.contact_sheet([(f"{r.imageId}: {r.question} -> {r.answer}"[:70], IM[r.imageId]) for r in GQ.head(6).itertuples()], 3, run.path("stimuli.png"))
    rmax = max(ranks)
    Qs = {L: lens.lens_subspace(lm, L, rmax) for L in layers}
    Qr = {L: lens.random_subspace(Qs[L].shape[0], rmax, 100 + L) for L in layers}
    Qpp = {L: lens.jpp_subspace(lm, L, rmax) for L in layers} if lm.jpp is not None else None
    for L in layers:  # Q[:, :r] must be the top r
        lens.assert_descending(lm, L, Qs[L])
        if Qpp is not None:
            lens.assert_descending(lm, L, Qpp[L], lm.jpp)
    Qu = {L: lens.unembed_subspace(lm, rmax) for L in layers}
    prompt = lambda q: lm.chat_media(["IMAGE", f"{q} Answer with one word."])  # noqa: E731
    prompt_noimg = lambda q: lm.chat([("user", f"Answer the question about a photo as best you can. {q} Answer with one word.")])  # noqa: E731
    texts = [prompt(r.question) for r in GQ.itertuples()]
    images = [IM[r.imageId] for r in GQ.itertuples()]
    # the clean run with readouts, and the image-token states for the PCA control
    rows_out, H_img = [], {L: [] for L in layers}
    for iid in pick:  # one image at a time: its questions share the token count
        sel = np.flatnonzero((GQ.imageId == iid).values)
        n_image = len([t for t in lm.encode([texts[sel[0]]], [images[sel[0]]])["input_ids"][0].tolist() if t == image_id])
        rj, rl, rp, _ = lens.ranks_at(lm, [texts[i] for i in sel], [images[i] for i in sel], where_image(n_image), n_image, [GQ.ans_ids.iloc[i] for i in sel], chunk=8)
        _, R = lm.forward([texts[i] for i in sel], [images[i] for i in sel], layers, where=where_image(n_image))
        for L in layers:
            H_img[L].append(R[L][:, :-1].reshape(-1, R[L].shape[-1]).numpy())
        for j, i in enumerate(sel):
            rows_out.append(
                dict(
                    i=int(i),
                    img_rank_j=int(rj[j, :, :-1, j].min()),
                    img_rank_l=int(rl[j, :, :-1, j].min()),
                    img_rank_p=int(rp[j, :, :-1, j].min()),
                    dec_rank_j=int(rj[j, :, -1, j].min()),
                    dec_rank_l=int(rl[j, :, -1, j].min()),
                    dec_rank_p=int(rp[j, :, -1, j].min()),
                    n_image=n_image,
                )
            )
    RO = pd.DataFrame(rows_out).set_index("i").sort_index()
    Qpca = {L: pca_subspace(np.concatenate(H_img[L]), rmax) for L in layers}

    def energy(Q, r):
        """mean share of the image-token states' squared norm inside the first r directions of Q, over the read layers"""
        out = []
        for L in layers:
            X = torch.as_tensor(np.concatenate(H_img[L]), dtype=torch.float32, device=lm.device)
            Qd = torch.as_tensor(Q[L][:, :r], dtype=torch.float32, device=lm.device)
            out.append(float(((X @ Qd) ** 2).sum(1).mean() / (X**2).sum(1).mean()))
        return float(np.mean(out))

    ENERGY = {f"J r={r}": energy(Qs, r) for r in ranks} | {f"random r={r}": energy(Qr, r) for r in ranks}
    ENERGY |= {f"PCA r={r}": energy(Qpca, r) for r in ranks} | {f"W_U r={r}": energy(Qu, r) for r in ranks}
    if Qpp is not None:
        ENERGY.update({f"J++ r={r}": energy(Qpp, r) for r in ranks})
    print("energy share of the image-token states removed by each ablation: " + ", ".join(f"{k} {v:.3f}" for k, v in ENERGY.items()))
    OVERLAP = {}  # how much of the lens subspace lies inside the other spans (chance: r / d)
    for r in ranks:
        OVERLAP[f"J in PCA r={r}"] = float(np.mean([np.linalg.norm(Qpca[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in layers]))
        OVERLAP[f"J in random r={r}"] = float(np.mean([np.linalg.norm(Qr[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in layers]))
        OVERLAP[f"J in W_U r={r}"] = float(np.mean([np.linalg.norm(Qu[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in layers]))
        if Qpp is not None:
            OVERLAP[f"J in J++ r={r}"] = float(np.mean([np.linalg.norm(Qpp[L][:, :r].T @ Qs[L][:, :r]) ** 2 / r for L in layers]))
    print("subspace overlap: " + ", ".join(f"{k} {v:.3f}" for k, v in OVERLAP.items()))
    MU = {L: np.concatenate(H_img[L]).mean(0) for L in layers}  # the mean image-token state per read layer (mean-ablation)
    H_img = None  # free the states

    def ablate(Q, part="lens", mu=None):
        return hooks.ablate_subspace(Q, at_image, layers, part=part, mu=mu)

    arms = [("clean", None, None), ("no image", None, "noimg")]
    for r in ranks:
        arms += [
            (f"ablate J part r={r}", ablate({L: Qs[L][:, :r] for L in layers}), None),
            (f"ablate random part r={r}", ablate({L: Qr[L][:, :r] for L in layers}), None),
            (f"ablate PCA part r={r}", ablate({L: Qpca[L][:, :r] for L in layers}), None),
            (f"ablate W_U part r={r}", ablate({L: Qu[L][:, :r] for L in layers}), None),
        ]
        if Qpp is not None:
            arms.append((f"ablate J++ part r={r}", ablate({L: Qpp[L][:, :r] for L in layers}), None))
    arms.append(("keep only J part r=256", ablate({L: Qs[L][:, :256] for L in layers}, "complement"), None))
    # content-specific, rank 1: the reference word's own lens direction against another question's answer word
    tok = [int(GQ.ans_ids.iloc[i][0]) for i in range(len(GQ))]
    oth = [next(tok[(i + j) % len(tok)] for j in range(1, len(tok)) if tok[(i + j) % len(tok)] != tok[i]) for i in range(len(tok))]
    Qtar = {L: lens.word_directions(lm, L, tok) for L in layers}
    Qoth = {L: lens.word_directions(lm, L, oth) for L in layers}
    arms += [
        ("ablate target word's lens direction", hooks.ablate_directions(Qtar, at_image, layers), None),
        ("ablate another word's lens direction", hooks.ablate_directions(Qoth, at_image, layers), None),
    ]
    if not args.no_mean_ablation:
        arms += [
            ("mean-ablate J part r=256", ablate({L: Qs[L][:, :256] for L in layers}, mu=MU), None),
            ("mean-ablate random part r=256", ablate({L: Qr[L][:, :256] for L in layers}, mu=MU), None),
            ("mean-ablate PCA part r=256", ablate({L: Qpca[L][:, :256] for L in layers}, mu=MU), None),
            ("mean-ablate W_U part r=256", ablate({L: Qu[L][:, :256] for L in layers}, mu=MU), None),
        ]
    res_rows = []
    for name, hk, mode in arms:
        tx = [prompt_noimg(r.question) for r in GQ.itertuples()] if mode == "noimg" else texts
        im = None if mode == "noimg" else images
        lg, _ = lm.forward(tx, im, [], hooks=hk)
        pr = torch.log_softmax(lg.float(), -1)
        logp = np.array([float(torch.logsumexp(pr[i, GQ.ans_ids.iloc[i]], 0)) for i in range(len(GQ))])
        gen = lm.generate(tx, im if im is not None else [None] * len(tx), 6, hooks=hk)
        jv = judge.qa_match(GQ.question.tolist(), gen, GQ.answer.tolist())
        rx = noun_matches(gen, GQ.answer.str.lower().tolist(), set(GQ.answer.str.lower()))
        for i, r in enumerate(GQ.itertuples()):
            res_rows.append(
                dict(
                    arm=name,
                    i=i,
                    imageId=r.imageId,
                    qid=r.id,
                    bin=r.bin,
                    plen=int(r.plen),
                    question=r.question,
                    reference=r.answer,
                    generation=gen[i],
                    correct=jv[i],
                    correct_regex=bool(rx[i]),
                    logp_ref=logp[i],
                )
            )
        acc = np.nanmean([float(v) for v in jv if v is not None])
        print(f"{name:28s} accuracy (judge) {acc:.2f}  mean log p(ref) {logp.mean():.2f}  judge failures {sum(v is None for v in jv)}  [{time.time() - t0:.0f}s]")
    N1 = pd.DataFrame(res_rows).merge(RO, left_on="i", right_index=True)
    run.save_csv(N1, "gqa_ablation.csv")
    judge.label_sample(N1[N1.arm == "clean"], "gqa_ablation")

    # ---- summary with image-grouped bootstraps (rules: RECORD.md 02:15)
    clean = N1[N1.arm == "clean"].set_index("i")
    out = dict(n_images=len(pick), n_questions=int(len(GQ)), judge=judge.model, by_bin={}, readout={})
    for b in list(BINS) + ["all"]:
        sel = (N1.bin == b) if b != "all" else np.ones(len(N1), bool)
        d = N1[sel]
        by = {}
        for name in d.arm.unique():
            x = d[d.arm == name].set_index("i")
            dlog = (x.logp_ref - clean.loc[x.index].logp_ref).values
            by[name] = dict(accuracy=boot_scene(x.correct.astype(float).fillna(0).values, x.imageId.values), dlogp=boot_scene(dlog, x.imageId.values))
        out["by_bin"][b] = by
        j = by.get("ablate J part r=256", {}).get("dlogp", (np.nan,) * 3)
        rnd = by.get("ablate random part r=256", {}).get("dlogp", (np.nan,) * 3)
        print(f"bin {b:3s}: acc clean {by['clean']['accuracy'][0]:.2f} no-image {by['no image']['accuracy'][0]:.2f} | dlogp: J {fmt(j)}  random {fmt(rnd)}")
    for key, lab in (
        ("img_rank_j", "J-lens on image tokens"),
        ("img_rank_l", "logit lens on image tokens"),
        ("img_rank_p", "J++ on image tokens"),
        ("dec_rank_j", "J-lens at the decision position"),
        ("dec_rank_l", "logit lens at the decision position"),
        ("dec_rank_p", "J++ at the decision position"),
    ):
        v = clean[key].values
        out["readout"][lab] = boot_scene(lens.readable(v), clean.imageId.values) if (v >= 1).any() else None
    # the same, restricted to the questions the model answers correctly with the image
    ok_i = clean.index[clean.correct == True]  # noqa: E712 - verdicts can be None
    strat = {}
    for name in N1.arm.unique():
        x = N1[(N1.arm == name) & (N1.i.isin(ok_i))].set_index("i")
        if len(x):
            strat[name] = dict(
                accuracy=boot_scene(x.correct.astype(float).fillna(0).values, x.imageId.values),
                dlogp=boot_scene((x.logp_ref - clean.loc[x.index].logp_ref).values, x.imageId.values),
                n=int(len(x)),
            )
    out["clean_correct_stratified"] = strat
    a = out["by_bin"]["all"]
    out["evidence_dlogp_clean_minus_noimage"] = boot_scene((clean.logp_ref - N1[N1.arm == "no image"].set_index("i").loc[clean.index].logp_ref).values, clean.imageId.values)
    out["energy_removed"] = ENERGY
    out["subspace_overlap"] = OVERLAP
    ev = out["evidence_dlogp_clean_minus_noimage"]
    necessary = ev[0] >= 1.0 and ev[1] > 0  # the evidence gate: the image matters for the answer
    jd, pd_ = a["ablate J part r=256"]["dlogp"], a["ablate PCA part r=256"]["dlogp"]
    xj = N1[N1.arm == "ablate J part r=256"].set_index("i").logp_ref
    xr = N1[N1.arm == "ablate random part r=256"].set_index("i").logp_ref
    dd = boot_scene((xj - xr).values, clean.imageId.values)
    out["J_minus_random_r256"] = dd
    out["verdict"] = (
        "evidence not necessary (no-image gate failed)"
        if not necessary
        else "the J-readable part is specifically necessary"
        if dd[0] <= -0.5 and dd[2] < 0 and jd[0] <= pd_[0]
        else "the J-readable part is not specifically necessary (random/PCA ablations hurt as much)"
        if dd[2] > -0.5 and dd[1] > -1.0
        else "undetermined at this n"
    )
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")
    print(f"evidence: clean - no-image log p(ref) = {fmt(ev)} nats; J - random (r = 256) {fmt(dd)} nats -> {out['verdict']}")


if __name__ == "__main__":
    main()
