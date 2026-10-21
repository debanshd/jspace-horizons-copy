"""Head-output patching: is one reader head's output enough to change the answer? (R8 in RECORD.md; the sufficiency half of the circuit)

The donor run sees object X in slot q with the same question as the receiver, which sees Z. At one softmax layer L the attention
module's per-head outputs (the input of o_proj, [heads x head_dim] per position) are recorded at the answer position (the last prompt
token) in the donor run, and written into the receiver at the same position: one head at a time, all heads of the layer together,
and a random head not among the readers. Per scene: the answer, follows X, keeps Z, and ld = log p(X's answer) - log p(Z's answer).

Rule (frozen 21 Oct 08:00): a head is a sufficient reader for a question when patching its output alone makes the answer follow X on
>= 0.5 of scenes; the layer's readers are sufficient together when patching every head of the layer does. Prediction (2B): animal
follows X on >= 0.5 with all of layer 7's heads, and heads 1 and 3 alone each reach 0.25 to 0.5; say follows X on >= 0.5 with all
of layer 19 (or 15), heads 1 and 5 alone 0.2 to 0.5; a non-reader head moves nothing (<= 0.05).

Outputs in runs/<run>/<model>/: head_patch.csv, summary.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_knockout import attention_layers  # noqa: E402
from read_battery import INTRO, QUESTIONS, first_token_ids, matches, truth  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN


@torch.no_grad()
def run_with_head_patch(lm, texts, images, n_new, mod, H, head_dim, donor_out, heads, ids_x, ids_z, positions="text"):
    """generate with the o_proj input at the last prompt position replaced, for the listed heads, by donor_out[i] ([H * head_dim])"""

    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        b, s = enc["input_ids"].shape
        Ds = [torch.as_tensor(donor_out[i], device=enc["input_ids"].device) for i in idx]  # [K_i, H * head_dim] each
        sel = torch.zeros(H * head_dim, dtype=torch.bool, device=enc["input_ids"].device)
        for h in heads:
            sel[h * head_dim : (h + 1) * head_dim] = True
        last_only = positions == "last"

        def pre(m, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"] if "hidden_states" in kwargs else None
            if x is None or x.shape[1] == 1:
                return None
            x = x.clone()
            for r, D in enumerate(Ds):  # left padding: the suffix of each row is its text after the image, ending at the answer position
                k = 1 if last_only else D.shape[0]
                x[r, -k:, sel] = D[-k:, sel].to(x.dtype)
            return ((x,) + tuple(args[1:]), kwargs) if args else (args, dict(kwargs, hidden_states=x))

        handle = mod.o_proj.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            g = lm.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            handle.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
        return [
            (lm.tokenizer.decode(g.sequences[j, s:], skip_special_tokens=True).strip(), float(torch.logsumexp(lp[j, ids_x[i]], 0)), float(torch.logsumexp(lp[j, ids_z[i]], 0)))
            for j, i in enumerate(idx)
        ]

    return lm.run_batches(texts, images, fn)


@torch.no_grad()
def capture_oproj_input(lm, texts, images, mod):
    """the o_proj input at the last prompt position, per prompt ([n, H * head_dim])"""
    out = [None] * len(texts)

    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        store = {}

        ids = enc["input_ids"]
        K = [len(row) - 1 - max(j for j, t in enumerate(row.tolist()) if t == lm.image_token_id) for row in ids]  # text tokens after the image

        def pre(m, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            store["x"] = [x[r, -K[r] :].float().cpu().numpy() for r in range(x.shape[0])]
            return None

        handle = mod.o_proj.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            lm._run(enc)
        finally:
            handle.remove()
        return store["x"]

    for i, r in zip(range(len(texts)), lm.run_batches(texts, images, fn)):
        out[i] = r
    return out


def main():
    ap = parser(__doc__, "head_patch")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--layers", default="7,11,15,19")
    ap.add_argument("--questions", default="animal,say")
    ap.add_argument("--positions", choices=["text", "last"], default="text", help="patch the heads' outputs at every text position after the image, or at the answer position only")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "head_patch")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank = crops.words, crops.tiles
    t0 = time.time()
    rng = np.random.default_rng(17)  # the battery's scenes
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
    don_img = [stimuli.scene(r.q, r.tx) for r in S.itertuples()]
    rec_img = [stimuli.scene(r.q, r.tz) for r in S.itertuples()]
    mods = dict(attention_layers(lm))
    cfg = lm.hf.config.get_text_config()
    H, head_dim = cfg.num_attention_heads, cfg.head_dim
    layers = [int(v) for v in args.layers.split(",")]
    res = []
    for task in args.questions.split(","):
        tmpl, needs, kind = QUESTIONS[task]
        TX = [truth(task, r, r.X) for r in S.itertuples()]
        TZ = [truth(task, r, r.Z) for r in S.itertuples()]
        ids_x = [first_token_ids(lm, t, kind) for t in TX]
        ids_z = [first_token_ids(lm, t, kind) for t in TZ]
        txt = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4])]) for r in S.itertuples()]
        for L in layers:
            mod = mods[L]
            donor_out = capture_oproj_input(lm, txt, don_img, mod)
            arms = [("clean", [])] + [(f"head {h}", [h]) for h in range(H)] + [("all heads", list(range(H)))]
            for name, heads in arms:
                if name == "clean":
                    out = run_with_head_patch(lm, txt, rec_img, 5, mod, H, head_dim, donor_out, [], ids_x, ids_z, args.positions)
                else:
                    out = run_with_head_patch(lm, txt, rec_img, 5, mod, H, head_dim, donor_out, heads, ids_x, ids_z, args.positions)
                for i, r in enumerate(S.itertuples()):
                    gen, lpx, lpz = out[i]
                    res.append(
                        dict(
                            task=task,
                            layer=L,
                            arm=name,
                            head=(heads[0] if len(heads) == 1 else None),
                            scene=r.scene,
                            X=r.X,
                            Z=r.Z,
                            generation=gen,
                            follows_X=matches(task, gen, TX[i], words),
                            keeps_Z=matches(task, gen, TZ[i], words),
                            ld=lpx - lpz,
                            distinct=TX[i] != TZ[i],
                        )
                    )
                d_ = [x for x in res if x["task"] == task and x["layer"] == L and x["arm"] == name and x["distinct"]]
                print(f"{task:7s} layer {L:2d} {name:10s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  ld {np.mean([x['ld'] for x in d_]):+.2f}  [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(res)
    run.save_csv(R, "head_patch.csv")
    out = dict(n=int(len(S)), layers=layers, heads=H, head_dim=head_dim, questions={})
    for task in args.questions.split(","):
        out["questions"][task] = {}
        d = R[(R.task == task) & R.distinct]
        scenes = np.sort(d.scene.unique())
        for L in layers:
            e = {}
            for name in d[d.layer == L].arm.unique():
                g = d[(d.layer == L) & (d.arm == name)].set_index("scene").reindex(scenes)
                e[name] = dict(follows_X=boot_scene(g.follows_X.astype(float).values, scenes), ld=boot_scene(g.ld.values, scenes))
            e["sufficient_heads"] = [name for name in e if name.startswith("head") and e[name]["follows_X"][0] >= 0.5]
            out["questions"][task][L] = e
    print(
        "head patch: "
        + "; ".join(
            f"{t} L{L}: all heads {v['all heads']['follows_X'][0]:.2f}, best head "
            + max((k for k in v if k.startswith("head")), key=lambda k: v[k]["follows_X"][0])
            + f" {max(v[k]['follows_X'][0] for k in v if k.startswith('head')):.2f}"
            for t, dd in out["questions"].items()
            for L, v in dd.items()
        )
    )
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
