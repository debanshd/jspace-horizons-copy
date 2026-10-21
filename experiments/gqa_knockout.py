"""GQA knockout: do the two reading stations hold on natural photos with the benchmark's own questions? (R5 in RECORD.md)

GQA testdev-balanced photos. Three kinds of question per image, from GQA's own types: a name ("query" questions whose answer names an
object or category), an attribute ("query" questions about an attribute, "what color is the car"), and a decision ("verify" yes / no,
or "choose" between two options). Attention from every text position after the image to the image tokens is blocked at softmax layers
(single, from L on, up to L), as in attention_knockout.py, with no patching and so no mixed states. Per question and arm: the answer,
the judge's verdict, and log p(the reference answer's first token). Cost = clean log p minus the knockout's, in nats.

Rule (frozen 21 Oct 05:20): per kind, last reader = the largest softmax layer whose "from L on" knockout costs >= 1 nat with the
image-grouped interval above 0. "The stations hold on natural photos" when the decisions' last reader is at or before the model's
decision station (layer 11 on the 4B, 7 on the 2B) and the names' last reader is at or after the copy station (19); otherwise
"the canvas stations do not transfer". Attributes are reported with no rule (prediction: with the names).

Outputs in runs/<run>/<model>/: gqa_knockout.csv, summary.json, stimuli.png, gqa_knockout_to_label.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import Knockout, attention_layers  # noqa: E402

from jspace import hooks, stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import RESULTS_DIR, require_data  # noqa: E402
from jspace.judge import Judge  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

COLORS = {"white", "black", "red", "blue", "green", "yellow", "brown", "gray", "grey", "orange", "pink", "purple", "silver", "gold", "tan", "beige", "dark", "light", "blond", "blonde"}


@torch.no_grad()
def generate_logp(lm, texts, images, n_new, ko, ids, hooks_=None):
    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        if ko is not None:
            ko.hooks(enc, idx)
        handles = [lm.register(L, f) for L, f in (hooks_(enc, idx) if hooks_ else [])]
        try:
            g = lm.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            if ko is not None:
                ko.remove()
            for h in handles:
                h.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
        return [(lm.tokenizer.decode(g.sequences[j, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip(), float(torch.logsumexp(lp[j, ids[i]], 0))) for j, i in enumerate(idx)]

    return lm.run_batches(texts, images, fn)


def main():
    ap = parser(__doc__, "gqa_knockout")
    ap.add_argument("--n-images", type=int, default=48)
    ap.add_argument(
        "--mode",
        choices=["knockout", "patch"],
        default="knockout",
        help="patch: write a blank image's token states into the photo's tokens from layer L on, for every L (reaches the linear-attention layers)",
    )
    args = ap.parse_args()
    require_data(args.data_dir, "gqa")
    run = Run(args, "gqa_knockout")
    args.run = args.run if args.mode == "knockout" else args.run
    lm = Session(args.model)
    run.write_config(lm)
    t0 = time.time()
    gqa = args.data_dir / "gqa"
    gq = pd.read_parquet(gqa / "testdev_balanced_instructions" / "testdev-00000-of-00001.parquet")
    gq["structural"] = gq.types.apply(lambda t: t["structural"])
    gq["sem"] = gq.types.apply(lambda t: t["semantic"])
    gq = gq[gq.answer.str.split().str.len() == 1].copy()
    gq["ans_ids"] = gq.answer.apply(lambda a: lm.single_ids([a, " " + a, a.capitalize(), " " + a.capitalize()]))
    gq = gq[gq.ans_ids.apply(len) > 0]

    def kind(r):
        if r.structural == "query" and r.sem in ("cat", "rel") and r.answer.lower() not in COLORS:
            return "name"
        if r.structural == "query" and r.sem == "attr":
            return "attribute"
        if r.structural in ("verify", "choose"):
            return "decision"
        return None

    gq["kind"] = [kind(r) for r in gq.itertuples()]
    gq = gq[gq.kind.notna()]
    rng = np.random.default_rng(0)
    imgs_ok = [i for i, g in gq.groupby("imageId") if g.kind.nunique() == 3]
    pick = [imgs_ok[j] for j in rng.permutation(len(imgs_ok))[: args.n_images]]
    rows_q = []
    for iid in pick:
        g = gq[gq.imageId == iid]
        for k in ("name", "attribute", "decision"):
            gk = g[g.kind == k]
            rows_q.append(gk.iloc[int(rng.integers(len(gk)))])
    GQ = pd.DataFrame(rows_q).reset_index(drop=True)
    imtab = pd.read_parquet(gqa / "testdev_balanced_images" / "testdev-00000-of-00001.parquet")
    imtab = imtab[imtab.id.isin(pick)].set_index("id")
    IM = {iid: stimuli.fit_photo(imtab.loc[iid, "image"]["bytes"]) for iid in pick}
    dec = GQ[GQ.kind == "decision"]
    print(f"GQA: {len(pick)} images x 3 kinds; decisions {dec.structural.value_counts().to_dict()}, answers {dec.answer.value_counts().head(4).to_dict()}  [{time.time() - t0:.0f}s]")
    stimuli.contact_sheet([(f"{r.imageId}: {r.question} -> {r.answer}"[:70], IM[r.imageId]) for r in GQ.head(6).itertuples()], 3, run.path("stimuli.png"))
    texts = [lm.chat_media(["IMAGE", r.question + " Answer with one word."]) for r in GQ.itertuples()]
    images = [IM[r.imageId] for r in GQ.itertuples()]
    ids = GQ.ans_ids.tolist()
    AL = [i for i, _ in attention_layers(lm)]
    judge = Judge(run.dir, reference_caches=[RESULTS_DIR / "judge_cache.json"])
    res = []
    if args.mode == "patch":
        # the blank donor: a grey image of the receiver's size, so it has the same image tokens; its states at every layer
        LF = lm.all_layers()
        blanks = [Image.new("RGB", im.size, (128, 128, 128)) for im in images]
        n_img = [sum(1 for t in lm.encode([texts[j]], [images[j]])["input_ids"][0].tolist() if t == lm.image_token_id) for j in range(len(texts))]
        P = max(n_img)
        REC = {L: np.zeros((len(texts), P, lm.d_model), np.float16) for L in LF}
        for j in range(len(texts)):
            _, Rb = lm.forward([lm.chat_media(["IMAGE", "Describe the image."])], [blanks[j]], LF, where=lambda ids: [k for k, t in enumerate(ids) if t == lm.image_token_id])
            for L in LF:
                REC[L][j, : n_img[j]] = Rb[L][0].half().numpy()
        print(f"blank donors captured ({len(LF)} layers; image tokens {min(n_img)} to {max(n_img)})  [{time.time() - t0:.0f}s]")

        def pos_of(j, s_len, n=n_img, texts_=texts):
            ids = lm.encode([texts_[j]], [images[j]])["input_ids"][0].tolist()
            img = [k for k, t in enumerate(ids) if t == lm.image_token_id]
            return np.array([s_len - len(ids) + k for k in img] + [s_len - len(ids) + img[-1]] * (P - len(img)))

        pos = [(lambda s_len, _j=j: pos_of(_j, s_len)) for j in range(len(texts))]
        arms = [("clean", None, "clean", None)] + [(f"blank state from layer {L}", hooks.patch([L], REC, pos), "from", L) for L in LF]
    else:
        arms = [("no knockout", None, "clean", None)]
        arms += [(f"block layer {L}", [L], "single", L) for L in AL]
        arms += [(f"block layers {L} on", [M for M in AL if M >= L], "from", L) for L in AL]
        arms += [(f"block layers up to {L}", [M for M in AL if M <= L], "upto", L) for L in AL]
    for name, block, kind_, L in arms:
        if args.mode == "patch":
            out = generate_logp(lm, texts, images, 6, None, ids, hooks_=block)
        else:
            ko = Knockout(lm, block, lm.image_token_id) if block else None
            out = generate_logp(lm, texts, images, 6, ko, ids)
        gen = [o[0] for o in out]
        jv = judge.qa_match(GQ.question.tolist(), gen, GQ.answer.tolist())
        for i, r in enumerate(GQ.itertuples()):
            res.append(
                dict(
                    arm=name,
                    kind=kind_,
                    layer=L,
                    i=i,
                    imageId=r.imageId,
                    qkind=r.kind,
                    structural=r.structural,
                    sem=r.sem,
                    question=r.question,
                    reference=r.answer,
                    generation=gen[i],
                    correct=jv[i],
                    logp=out[i][1],
                )
            )
        d_ = pd.DataFrame([x for x in res if x["arm"] == name])
        print(
            f"{name:24s} "
            + "  ".join(f"{k}: correct {d_[d_.qkind == k].correct.astype(float).mean():.2f} logp {d_[d_.qkind == k].logp.mean():+.2f}" for k in ("name", "attribute", "decision"))
            + f"  [{time.time() - t0:.0f}s]"
        )
    R = pd.DataFrame(res)
    run.save_csv(R, "gqa_knockout.csv")
    judge.label_sample(R[R.arm == "no knockout"], "gqa_knockout")
    out = dict(n_images=len(pick), n_questions=int(len(GQ)), attention_layers=AL, judge=judge.model, judge_calls=judge.n_calls, judge_failed=judge.n_failed, kinds={})
    for k in ("name", "attribute", "decision"):
        d = R[R.qkind == k]
        clean = d[d.kind == "clean"].set_index("i")
        e = dict(n=int(len(clean)), clean_correct=float(clean.correct.astype(float).mean()), clean_logp=float(clean.logp.mean()))
        layers_ = sorted(int(x) for x in d[d.kind == "from"].layer.dropna().unique())
        for kind_ in ("single", "from", "upto"):
            e[kind_] = {}
            for L in layers_ if args.mode == "patch" else AL:
                if kind_ != "from" and args.mode == "patch":
                    continue
                g = d[(d.kind == kind_) & (d.layer == L)].set_index("i").reindex(clean.index)
                e[kind_][L] = dict(correct=boot_scene(g.correct.astype(float).fillna(0).values, g.imageId.values), cost=boot_scene((clean.logp - g.logp).values, g.imageId.values))
        e["readers"] = [L for L in e["single"] if e["single"][L]["cost"][0] >= 1 and e["single"][L]["cost"][1] > 0]
        e["last_reader"] = max([L for L in e["from"] if e["from"][L]["cost"][0] >= 1 and e["from"][L]["cost"][1] > 0], default=None)
        e["all_blocked_cost"] = e["from"][min(e["from"])]["cost"]
        out["kinds"][k] = e
    print(
        "GQA knockout: "
        + "; ".join(f"{k}: clean {v['clean_correct']:.2f}, readers {v['readers']}, last reader {v['last_reader']}, all blocked cost {v['all_blocked_cost'][0]:.1f}" for k, v in out["kinds"].items())
    )
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
