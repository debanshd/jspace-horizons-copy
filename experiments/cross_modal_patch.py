"""Cross-modal head-output interchange: is what a reader head delivers the same object whether the model saw the thing or read its
word? (R9 in RECORD.md; rubric gates XVIII, XIX, XXI, XLIV)

Four prompts per scene: the picture of X or of Z in slot q, and the typed sentence "Image slot q contains a X." or "... a Z." with
no picture, each followed by the same question. The donor holds X, the receiver Z. At the chosen softmax layers the donor's per-head
attention outputs (the input of o_proj) at the question's token positions, the longest common suffix of the two prompts' token ids
ending at the answer position, are written into the receiver: one head at a time, all heads of a layer, and joint patches over
several layers. Directions: typed donor into picture receiver, picture into typed, and the two same-modality references.

Alignment: for each head, the cosine between its output at the answer position for X seen and X typed, minus the cosine for X seen
and Z typed with the same question (within minus across), scene-bootstrapped; and the same from the typed side.

Rules (frozen 21 Oct 08:30): (a) a head's content is modality-free when the typed donor's output alone makes the picture receiver
follow X on >= 0.5 of scenes and the picture donor's makes the typed receiver follow X on >= 0.3, non-reader heads within 0.05 of
clean; (b) the copy's readers are jointly sufficient when all heads of 11, 15 and 19 give say >= 0.5 in the same-modality direction;
(c) a head carries the object when its within-minus-across cosine interval excludes 0.

Outputs in runs/<run>/<model>/: cross_modal_patch.csv (every arm and scene), alignment.csv (per head cosines), summary.json.
"""

import json
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
MODES = ("picture", "typed")


def scenes(words, bank, n):
    """the battery's scenes (read_battery.py, rng 17)"""
    rng = np.random.default_rng(17)
    rows = []
    for i in range(n):
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
    return pd.DataFrame(rows)


def prompts(lm, S, tmpl, mode, obj):
    """texts and images for one modality (picture or typed) holding X or Z"""
    texts, images = [], []
    for r in S.itertuples():
        question = tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4])
        word, tile = (r.X, r.tx) if obj == "X" else (r.Z, r.tz)
        if mode == "typed":
            texts.append(lm.chat([("user", INTRO + f" Image slot {r.q} contains a {word}. " + question)]))
            images.append(None)
        else:
            texts.append(lm.chat_media([INTRO, "IMAGE", question]))
            images.append(stimuli.scene(r.q, tile))
    return texts, images


def common_suffix(lm, a, b):
    """length of the longest common suffix of two prompts' token ids (the shared question, ending at the answer position)"""
    ia = lm.tokenizer(a, add_special_tokens=False)["input_ids"]
    ib = lm.tokenizer(b, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ia), len(ib)) and ia[-1 - k] == ib[-1 - k]:
        k += 1
    return k


@torch.no_grad()
def capture(lm, texts, images, mods, K):
    """per layer, the o_proj input at the last K[i] positions of each prompt ([K_i, H * head_dim], float32 on the cpu)"""
    out = {L: [None] * len(texts) for L in mods}

    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        store = {}
        handles = []
        for L, mod in mods.items():

            def pre(m, args, kwargs, L=L):
                x = args[0] if args else kwargs["hidden_states"]
                store[L] = [x[r, -K[i] :].float().cpu().numpy() for r, i in enumerate(idx)]
                return None

            handles.append(mod.o_proj.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            lm._run(enc)
        finally:
            for h in handles:
                h.remove()
        return [{L: store[L][r] for L in mods} for r in range(len(idx))]

    for i, r in enumerate(lm.run_batches(texts, images, fn)):
        for L in mods:
            out[L][i] = r[L]
    return out


@torch.no_grad()
def generate_patched(lm, texts, images, n_new, mods, head_dim, spec, donor, ids_x, ids_z):
    """generate with the o_proj input at the question's positions replaced, for spec = {layer: [heads]}, by the donor's ([K_i, H * head_dim])"""

    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        s = enc["input_ids"].shape[1]
        dev = enc["input_ids"].device
        handles = []
        for L, heads in spec.items():
            hd = head_dim[L] if isinstance(head_dim, dict) else head_dim
            sel = torch.zeros(donor[L][idx[0]].shape[1], dtype=torch.bool, device=dev)
            for h in heads:
                sel[h * hd : (h + 1) * hd] = True
            Ds = [torch.as_tensor(donor[L][i], device=dev) for i in idx]

            def pre(m, args, kwargs, sel=sel, Ds=Ds):
                x = args[0] if args else kwargs["hidden_states"] if "hidden_states" in kwargs else None
                if x is None or x.shape[1] == 1:  # decoding steps
                    return None
                x = x.clone()
                for r, D in enumerate(Ds):  # left padding: the suffix of each row is the question, ending at the answer position
                    k = D.shape[0]
                    x[r, -k:, sel] = D[:, sel].to(x.dtype)
                return ((x,) + tuple(args[1:]), kwargs) if args else (args, dict(kwargs, hidden_states=x))

            handles.append(mods[L].o_proj.register_forward_pre_hook(pre, with_kwargs=True))
        try:
            g = lm.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
        return [
            (lm.tokenizer.decode(g.sequences[j, s:], skip_special_tokens=True).strip(), float(torch.logsumexp(lp[j, ids_x[i]], 0)), float(torch.logsumexp(lp[j, ids_z[i]], 0)))
            for j, i in enumerate(idx)
        ]

    return lm.run_batches(texts, images, fn)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def head_geometry(mods):
    """(heads, {layer: head_dim}) read from the attention modules, since Gemma-4 keeps head_dim per layer and its config refuses a global value"""
    dims = {}
    for L, mod in mods.items():
        hd = getattr(mod, "head_dim", None)
        if hd is None:
            hd = mod.o_proj.in_features // getattr(mod, "num_heads", mod.config.num_attention_heads)
        dims[L] = int(hd)
    heads = {L: mod.o_proj.in_features // dims[L] for L, mod in mods.items()}
    assert len(set(heads.values())) == 1, f"the layers differ in head count: {heads}"
    return next(iter(heads.values())), dims


def summarize(R, A, layers, H, head_dim):
    """the rules of 08:30 on the saved rows: sufficient arms per direction, and which heads carry the object across modalities"""
    out = dict(n=int(R.scene.nunique()), layers=layers, heads=H, head_dim=head_dim, directions=sorted(R.direction.unique()), questions={}, alignment={})
    for task in R.task.unique():
        out["questions"][task] = {}
        d = R[(R.task == task) & R.distinct]
        sc = np.sort(d.scene.unique())
        for direction in d.direction.unique():
            e = {}
            for name in d[d.direction == direction].arm.unique():
                g = d[(d.direction == direction) & (d.arm == name)].set_index("scene").reindex(sc)
                e[name] = dict(follows_X=boot_scene(g.follows_X.astype(float).values, sc), ld=boot_scene(g.ld.values, sc))
            e["sufficient"] = [name for name in e if name != "clean" and isinstance(e[name], dict) and e[name]["follows_X"][0] >= (0.3 if direction == "picture>typed" else 0.5)]
            out["questions"][task][direction] = e
        a = A[A.task == task]
        out["alignment"][task] = {}
        for L in layers:
            out["alignment"][task][L] = {}
            for h in range(H):
                g = a[(a.layer == L) & (a["head"] == h)].set_index("scene").reindex(sc)
                out["alignment"][task][L][h] = dict(
                    within=boot_scene(g.within.values, sc),
                    within_minus_across=boot_scene((g.within - g.across_from_picture).values, sc),
                    within_minus_across_typed=boot_scene((g.within - g.across_from_typed).values, sc),
                    same_modality_other=boot_scene(g.same_modality_other.values, sc),
                    # a modality component: same modality and the other object closer than the other modality and the other object
                    modality_component=boot_scene((g.same_modality_other - g.across_from_picture).values, sc),
                    modality_component_typed=boot_scene((g.typed_other - g.across_from_typed).values, sc),
                )
    for task, dd in out["questions"].items():
        for direction, e in dd.items():
            print(f"{task} {direction}: sufficient {e['sufficient']}")
    for task, dd in out["alignment"].items():
        for L, hh in dd.items():
            carriers = [h for h, v in hh.items() if v["within_minus_across"][1] > 0]
            print(f"{task} layer {L}: heads whose output carries the object across modalities {carriers}: " + ", ".join(f"{h} {hh[h]['within_minus_across'][0]:+.2f}" for h in carriers))
            tagged = [h for h, v in hh.items() if v["modality_component"][1] > 0.02]
            print(f"{task} layer {L}: heads with a modality component above 0.02 {tagged}: " + ", ".join(f"{h} {hh[h]['modality_component'][0]:+.2f}" for h in tagged))
    return out


def main():
    ap = parser(__doc__, "cross_modal_patch")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--layers", default="7,11,15,19")
    ap.add_argument("--joint", default="7+11,11+15+19,7+11+15+19", help="joint all-head patches over these layer groups")
    ap.add_argument("--questions", default="animal,say")
    ap.add_argument("--directions", default="typed>picture,picture>typed,picture>picture,typed>typed")
    ap.add_argument("--from-csv", type=Path, default=None, help="rebuild summary.json of a saved run folder from its CSVs, no model")
    ap.add_argument("--alignment-only", action="store_true", help="no patch arms: the per-head cosines at every shared position, to alignment_positions.csv")
    args = ap.parse_args()
    if args.from_csv:
        R, A = pd.read_csv(args.from_csv / "cross_modal_patch.csv"), pd.read_csv(args.from_csv / "alignment.csv")
        H = int(A["head"].max()) + 1
        prior = json.load(open(args.from_csv / "summary.json"))["head_dim"] if (args.from_csv / "summary.json").exists() else None
        out = summarize(R, A, sorted(int(L) for L in A.layer.unique()), H, prior)
        out["rebuilt_from_csv"] = True
        json.dump(out, open(args.from_csv / "summary.json", "w"), indent=2)
        return
    require_data(args.data_dir, "coco")
    run = Run(args, "cross_modal_patch")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    S = scenes(crops.words, crops.tiles, args.n)
    words = crops.words
    layers = [int(v) for v in args.layers.split(",")]
    joint = [[int(v) for v in g.split("+")] for g in args.joint.split(",")] if args.joint else []
    mods = {L: m for L, m in dict(attention_layers(lm)).items() if L in layers}
    H, head_dim = head_geometry(mods)
    directions = [tuple(d.split(">")) for d in args.directions.split(",")]
    t0 = time.time()
    res, align = [], []
    for task in args.questions.split(","):
        tmpl, needs, kind = QUESTIONS[task]
        TX = [truth(task, r, r.X) for r in S.itertuples()]
        TZ = [truth(task, r, r.Z) for r in S.itertuples()]
        ids_x = [first_token_ids(lm, t, kind) for t in TX]
        ids_z = [first_token_ids(lm, t, kind) for t in TZ]
        P = {(mode, obj): prompts(lm, S, tmpl, mode, obj) for mode in MODES for obj in ("X", "Z")}
        # the shared question: the common suffix of the picture and typed prompts, the same for X and Z (they differ only before it)
        K = [min(common_suffix(lm, P[("picture", "X")][0][i], P[("typed", "X")][0][i]), common_suffix(lm, P[("picture", "Z")][0][i], P[("typed", "Z")][0][i])) for i in range(len(S))]
        print(f"{task}: shared question suffix {min(K)} to {max(K)} tokens  [{time.time() - t0:.0f}s]")
        cap = {key: capture(lm, *P[key], mods, K) for key in P}
        if args.alignment_only:  # every shared position, counted from the answer position (0) backwards
            for L in layers:
                for h in range(H):
                    sl = slice(h * head_dim[L], (h + 1) * head_dim[L])
                    for i in range(len(S)):
                        cx, ct, cz, ctz = (cap[(m, o)][L][i][:, sl] for m, o in (("picture", "X"), ("typed", "X"), ("picture", "Z"), ("typed", "Z")))
                        for j in range(cx.shape[0]):
                            k = cx.shape[0] - 1 - j
                            w, a, o = cosine(cx[j], ct[j]), cosine(cx[j], ctz[j]), cosine(cx[j], cz[j])
                            align.append(dict(task=task, layer=L, head=h, scene=i, pos=k, within=w, across_from_picture=a, same_modality_other=o))
            continue
        # alignment at the answer position: within (X seen, X typed) minus across (X seen, Z typed), and from the typed side
        for L in layers:
            for h in range(H):
                sl = slice(h * head_dim[L], (h + 1) * head_dim[L])
                for i in range(len(S)):
                    px, tx = cap[("picture", "X")][L][i][-1, sl], cap[("typed", "X")][L][i][-1, sl]
                    pz, tz = cap[("picture", "Z")][L][i][-1, sl], cap[("typed", "Z")][L][i][-1, sl]
                    align.append(
                        dict(
                            task=task,
                            layer=L,
                            head=h,
                            scene=i,
                            within=cosine(px, tx),
                            across_from_picture=cosine(px, tz),
                            across_from_typed=cosine(tx, pz),
                            same_modality_other=cosine(px, pz),
                            typed_other=cosine(tx, tz),
                        )
                    )
        arms = [("clean", {})] + [(f"head {h} of {L}", {L: [h]}) for L in layers for h in range(H)] + [(f"all heads of {L}", {L: list(range(H))}) for L in layers]
        arms += [("all heads of " + "+".join(map(str, g)), {L: list(range(H)) for L in g}) for g in joint]
        for dmode, rmode in directions:
            donor = cap[(dmode, "X")]
            texts, images = P[(rmode, "Z")]
            for name, spec in arms:
                out = generate_patched(lm, texts, images, 5, mods, head_dim, spec, donor, ids_x, ids_z)
                for i, r in enumerate(S.itertuples()):
                    gen, lpx, lpz = out[i]
                    res.append(
                        dict(
                            task=task,
                            direction=f"{dmode}>{rmode}",
                            arm=name,
                            layers="+".join(map(str, spec)) if spec else "",
                            head=(list(spec.values())[0][0] if spec and len(spec) == 1 and len(list(spec.values())[0]) == 1 else None),
                            scene=r.scene,
                            X=r.X,
                            Z=r.Z,
                            K=K[i],
                            generation=gen,
                            follows_X=matches(task, gen, TX[i], words),
                            keeps_Z=matches(task, gen, TZ[i], words),
                            ld=lpx - lpz,
                            distinct=TX[i] != TZ[i],
                        )
                    )
                d_ = [x for x in res if x["task"] == task and x["direction"] == f"{dmode}>{rmode}" and x["arm"] == name and x["distinct"]]
                if name == "clean" or name.startswith("all"):
                    print(f"{task:7s} {dmode:>7s}>{rmode:<7s} {name:24s} follows X {np.mean([x['follows_X'] for x in d_]):.2f}  ld {np.mean([x['ld'] for x in d_]):+.2f}  [{time.time() - t0:.0f}s]")
    R, A = pd.DataFrame(res), pd.DataFrame(align)
    if args.alignment_only:
        run.save_csv(A, "alignment_positions.csv")
        g = A.groupby(["task", "layer", "head", "pos"]).agg(within_minus_across=("within", "mean"), across=("across_from_picture", "mean")).reset_index()
        g["within_minus_across"] -= g.pop("across")
        best = g.sort_values("within_minus_across", ascending=False).groupby(["task", "layer"]).head(3)
        print(best.to_string(index=False))
        return
    run.save_csv(R, "cross_modal_patch.csv")
    run.save_csv(A, "alignment.csv")
    out = summarize(R, A, layers, H, head_dim)
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
