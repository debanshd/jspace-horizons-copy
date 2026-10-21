"""Attention knockout: which layers read the object's image tokens for each question? (R2 in RECORD.md)

The second method beside patching. The receiver canvas (object Z in slot q) is shown with a question, and attention from every
text position after the image (the question and the answer position) to the image tokens is blocked at chosen layers, by editing
the attention mask of those layers' softmax attention. Nothing else changes: the image tokens still attend among themselves and
the text still attends to the text. Arms per question: no knockout; every attention layer alone; and cumulative sets, all layers
from L on and all layers up to L. In Qwen3.5 only the one layer in four with standard attention can be blocked (the linear
attention layers read through a recurrent state), so the arms range over those layers and the all-layers knockout is the control
that says whether the standard-attention layers carry the read at all. In Gemma-4 every layer can be blocked.

Per arm and scene the script saves the answer, whether it is still correct, and the log-probability of the correct answer's first
token against the alternative answer (the other kind's answer, as in the read battery). A question's reader layers are those
whose single-layer knockout costs log-probability, and the "from L on" curve gives the last layer that still reads.

Outputs in runs/<run>/<model>/: attention_knockout.csv, summary.json, crops.json.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_sweep import word_positions  # noqa: E402
from read_battery import QUESTIONS, TWO_BY_TWO, first_token_ids, matches, truth, with_distractors  # noqa: E402

from jspace import stimuli  # noqa: E402
from jspace.analysis import boot_scene  # noqa: E402
from jspace.config import require_data  # noqa: E402
from jspace.model import Session  # noqa: E402
from jspace.runs import Run, parser  # noqa: E402

ANIMALS, RIDDEN = stimuli.ANIMALS, stimuli.RIDDEN
INTRO = "Study the scene. It has an image grid with numbered slots."


def attention_layers(lm):
    """the decoder layers whose token mixer is softmax attention, with the attention module of each"""
    out = []
    for i, layer in enumerate(lm.model.layers):
        mod = getattr(layer, "self_attn", None)
        if mod is not None and getattr(layer, "block_type", "full_attention") != "linear_attention":
            out.append((i, mod))
    return out


class Knockout:
    """blocks attention from the text positions after the image to the image tokens at the given layers, for one batch at a time"""

    def __init__(self, lm, layers_to_block, image_token_id, head=None, all_but=False, key_positions=None, query_positions=None, query_all_but=False):
        self.lm, self.block, self.image_token_id = lm, set(layers_to_block), image_token_id
        self.head, self.all_but = head, all_but  # head-level: block one query head only, or every head except it
        self.key_positions = key_positions  # per prompt, the token positions to block instead of the image tokens (typed sentences)
        self.query_positions, self.query_all_but = query_positions, query_all_but  # per prompt, block the reads of these query tokens only (or of every other text token)
        self.handles = []
        self.mods = dict(attention_layers(lm))

    def hooks(self, enc, idx):
        self.remove()
        ids, am = enc["input_ids"], enc["attention_mask"]
        b, s = ids.shape
        dev = ids.device
        if self.key_positions is not None:  # the typed sentence: the object's word token(s), shifted past each row's left padding
            is_img = torch.zeros_like(ids, dtype=torch.bool)
            for r in range(b):
                n = int(am[r].sum())
                for p in self.key_positions[int(idx[r])]:
                    is_img[r, s - n + p] = True
        else:
            is_img = ids == self.image_token_id  # [b, s]
        last_img = torch.where(is_img, torch.arange(s, device=dev)[None, :], torch.full((b, s), -1, device=dev)).max(1).values  # [b]
        after = torch.arange(s, device=dev)[None, :] > last_img[:, None]  # text positions after the image
        if self.query_positions is not None:  # only some query tokens lose their reads (positions in the unpadded prompt, shifted past the padding)
            sel_q = torch.zeros_like(ids, dtype=torch.bool)
            for r in range(b):
                n = int(am[r].sum())
                for p in self.query_positions[int(idx[r])]:
                    sel_q[r, s - n + p] = True
            after = after & (~sel_q if self.query_all_but else sel_q)
        KO = after[:, :, None] & is_img[:, None, :]  # [b, q, k]: query after the image, key an image token
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev))[None] & am.bool()[:, None, :]  # [b, q, k] with padding keys off
        self.KO, self.causal = KO[:, None], causal[:, None]  # [b, 1, q, k]
        if self.head is not None:  # [b, H, q, k]: the knockout applies to one query head (or to all the others)
            H = self.lm.hf.config.get_text_config().num_attention_heads
            sel = torch.zeros(H, dtype=torch.bool, device=dev)
            sel[self.head] = True
            if self.all_but:
                sel = ~sel
            self.KO = self.KO & sel[None, :, None, None]

        def pre(mod, args, kwargs):
            m = kwargs.get("attention_mask")
            q = kwargs["hidden_states"].shape[1]
            if q == 1:  # a generated token: its reads are not blocked (the prompt pass decided the answer)
                return None
            if m is None:
                new = self.causal & ~self.KO
            elif m.dtype == torch.bool:
                new = m & ~self.KO[..., : m.shape[-2], : m.shape[-1]]
            else:
                new = m.masked_fill(self.KO[..., : m.shape[-2], : m.shape[-1]], torch.finfo(m.dtype).min)
            kwargs = dict(kwargs, attention_mask=new)
            return args, kwargs

        for L in self.block:
            self.handles.append(self.mods[L].register_forward_pre_hook(pre, with_kwargs=True))
        return []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def generate_scored(lm, texts, images, n_new, ko, ids_a, ids_b):
    def fn(idx):
        enc = lm.encode([texts[i] for i in idx], [images[i] for i in idx])
        if ko is not None:
            ko.hooks(enc, idx)
        try:
            g = lm.hf.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=lm.tokenizer.pad_token_id, output_scores=True, return_dict_in_generate=True)
        finally:
            if ko is not None:
                ko.remove()
        lp = torch.log_softmax(g.scores[0].float(), -1).cpu()
        res = []
        for j, i in enumerate(idx):
            text = lm.tokenizer.decode(g.sequences[j, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            res.append((text, float(torch.logsumexp(lp[j, ids_a[i]], 0)), float(torch.logsumexp(lp[j, ids_b[i]], 0))))
        return res

    return lm.run_batches(texts, images, fn)


def main():
    ap = parser(__doc__, "attention_knockout")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    ap.add_argument("--distractors", type=int, default=0, help="other objects' crops in other slots (R12, the load test)")
    ap.add_argument("--questions", default="2x2")
    ap.add_argument("--heads", default="", help="comma list of layers: add single-head and all-but-one-head knockouts at these layers")
    ap.add_argument("--layers", default="", help="comma list: block only these attention layers (default: every softmax-attention layer)")
    ap.add_argument("--typed", action="store_true", help="the typed sentence instead of the picture: block attention to the object's word token")
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "attention_knockout")
    lm = Session(args.model)
    if "flash" in str(getattr(lm.hf.config, "_attn_implementation", "")):
        raise SystemExit("the knockout needs an attention implementation that takes a mask (sdpa or eager)")
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank = crops.words, crops.tiles
    tasks = list(QUESTIONS) if args.questions == "all" else TWO_BY_TWO if args.questions == "2x2" else args.questions.split(",")
    t0 = time.time()
    attn = attention_layers(lm)
    AL = [i for i, _ in attn]
    if args.layers:
        keep = {int(v) for v in args.layers.split(",")}
        AL = [L for L in AL if L in keep]
    print(f"attention layers that can be blocked: {AL} of {lm.final + 1}")

    rng = np.random.default_rng(17)  # the receivers of the read battery: the same scenes
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
        rows.append(dict(scene=i, X=x, Z=z, q=q, set5=set5, tz=bank[z][(i + 1) % len(bank[z])]))
    S = pd.DataFrame(rows)
    rec_img = [with_distractors(r, r.tz, args.distractors, bank, words) for r in S.itertuples()]

    arms = [("no knockout", None, "clean", None)]
    arms += [(f"block layer {L}", [L], "single", L) for L in AL]
    arms += [(f"block layers {L} on", [M for M in AL if M >= L], "from", L) for L in AL]
    arms += [(f"block layers up to {L}", [M for M in AL if M <= L], "upto", L) for L in AL]
    H = lm.hf.config.get_text_config().num_attention_heads
    head_layers = [int(x) for x in args.heads.split(",") if x]
    for L in head_layers:
        arms += [(f"block head {h} of layer {L}", ([L], h, False), "head", L) for h in range(H)]
        arms += [(f"block all heads but {h} of layer {L}", ([L], h, True), "allbut", L) for h in range(H)]
    res = []
    for task in tasks:
        tmpl, needs, kind = QUESTIONS[task]
        TZ = [truth(task, r, r.Z) for r in S.itertuples()]
        TX = [truth(task, r, r.X) for r in S.itertuples()]
        ids_z = [first_token_ids(lm, t, kind) for t in TZ]
        ids_x = [first_token_ids(lm, t, kind) for t in TX]
        if args.typed:
            txt = [lm.chat([("user", INTRO + f" Image slot {r.q} contains a {r.Z}. " + tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4]))]) for r in S.itertuples()]
            keys = [word_positions(lm, t, r.Z, t.index("contains a ")) for t, r in zip(txt, S.itertuples())]
            imgs = [None] * len(S)
        else:
            txt = [lm.chat_media([INTRO, "IMAGE", tmpl.format(q=r.q, a=r.set5[0], b=r.set5[1], c=r.set5[2], d=r.set5[3], e=r.set5[4])]) for r in S.itertuples()]
            keys, imgs = None, rec_img
        for name, block, kind_, L in arms:
            ko = (
                (
                    Knockout(lm, *block[:1], lm.image_token_id, head=block[1], all_but=block[2], key_positions=keys)
                    if isinstance(block, tuple)
                    else Knockout(lm, block, lm.image_token_id, key_positions=keys)
                )
                if block
                else None
            )
            out = generate_scored(lm, txt, imgs, 5, ko, ids_z, ids_x)
            for i, r in enumerate(S.itertuples()):
                gen, lpz, lpx = out[i]
                res.append(
                    dict(
                        task=task,
                        needs=needs,
                        answer_type=kind,
                        arm=name,
                        kind=kind_,
                        layer=L,
                        head=(block[1] if isinstance(block, tuple) else None),
                        scene=r.scene,
                        Z=r.Z,
                        X=r.X,
                        q=r.q,
                        truth_Z=TZ[i],
                        alt_X=TX[i],
                        distinct=TX[i] != TZ[i],
                        generation=gen,
                        correct=matches(task, gen, TZ[i], words),
                        lp_Z=lpz,
                        lp_X=lpx,
                        ld=lpz - lpx,
                    )
                )
            d_ = [x for x in res if x["task"] == task and x["arm"] == name and x["distinct"]]
            print(f"{task:9s} {name:24s} correct {np.mean([x['correct'] for x in d_]):.2f}  ld {np.mean([x['ld'] for x in d_]):+.2f}  [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(res)
    run.save_csv(R, "attention_knockout.csv")

    out = dict(n=int(len(S)), n_layers=int(lm.final + 1), attention_layers=AL, typed=bool(args.typed), questions={})
    for task in tasks:
        d = R[(R.task == task) & R.distinct]
        scenes = np.sort(d.scene.unique())
        clean = d[d.kind == "clean"].set_index("scene").reindex(scenes)
        e = dict(needs=QUESTIONS[task][1], answer_type=QUESTIONS[task][2], n=int(len(scenes)), clean_correct=float(clean.correct.mean()), clean_ld=boot_scene(clean.ld.values, scenes))
        for kind_ in ("single", "from", "upto"):
            e[kind_] = {}
            for L in AL:
                g = d[(d.kind == kind_) & (d.layer == L)].set_index("scene").reindex(scenes)
                e[kind_][L] = dict(correct=boot_scene(g.correct.astype(float).values, scenes), ld=boot_scene(g.ld.values, scenes), cost=boot_scene((clean.ld - g.ld).values, scenes))
        # readers: single-layer knockouts that cost >= 1 nat with the interval above 0; last reader: the largest L whose "from L on"
        # knockout still costs >= 1 nat with the interval above 0 (rule frozen in RECORD.md, 21 Oct)
        e["readers"] = [L for L in AL if e["single"][L]["cost"][0] >= 1 and e["single"][L]["cost"][1] > 0]
        e["all_blocked_correct"] = e["from"][AL[0]]["correct"][0]
        e["all_blocked_ld"] = e["from"][AL[0]]["ld"]
        e["last_reader"] = max([L for L in AL if e["from"][L]["cost"][0] >= 1 and e["from"][L]["cost"][1] > 0], default=None)
        for L in head_layers:  # head-level: the cost of blocking each head alone, and of keeping each head alone
            for kind_ in ("head", "allbut"):
                e[f"{kind_} {L}"] = {}
                for h in range(H):
                    g = d[(d.kind == kind_) & (d.layer == L) & (d["head"] == h)].set_index("scene").reindex(scenes)
                    e[f"{kind_} {L}"][h] = dict(correct=boot_scene(g.correct.astype(float).values, scenes), cost=boot_scene((clean.ld - g.ld).values, scenes))
        out["questions"][task] = e
    print("knockout: " + "; ".join(f"{t}: readers {v['readers']}, last reader {v['last_reader']}, all blocked correct {v['all_blocked_correct']:.2f}" for t, v in out["questions"].items()))
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
