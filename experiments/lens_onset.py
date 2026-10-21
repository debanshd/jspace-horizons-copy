"""Lens onset: at which layer does a lens first read an object's name off its image tokens, and does a stronger lens read it earlier?
(O1 in RECORD.md, re-measured with the J++ lens where Neuronpedia fitted one; the instrument check behind "the lens lags the reader".)

One COCO crop on a one-slot canvas under a neutral prompt, 64 fresh crops (crop set 1). At every layer the image tokens' residual
states are read by the J-lens, the J++ lens (where fitted) and the plain logit lens, both on the mean-pooled image state and on
every image token separately. The object's name (best single-token form) is ranked against the whole vocabulary; a null is another
object's name. Onset = the first layer at which the name is in the lens's top k on at least a share s of scenes, for k in {1, 5, 10, 20}
and s in {0.25, 0.5, 0.75}; per token, the share of scenes on which any token reads the name. The read depths come from the
battery (runs/run7/<model>/summary.json) and the knockout (runs/run7k/<model>/summary.json) when they exist.

Rule (RECORD.md 03:25 amendment, 21 Oct): "the lens lags the decision station" when the J++ onset (top 10, half the scenes, pooled)
is at or after the last single reader of the animal rule by knockout; "the stronger lens sees the name while the decisions still read
it" when the J++ onset is two or more softmax layers before it.

Outputs in runs/<run>/<model>/: lens_onset.csv (scene x layer x lens: rank of the name, rank of the null, pooled and best token),
summary.json (curves and onsets per lens and threshold), crops.json.
"""

import time

import numpy as np
import pandas as pd
import torch

from jspace import stimuli
from jspace.config import require_data
from jspace.model import Session
from jspace.runs import Run, parser

INTRO = "Study the scene."
TOPS = (1, 5, 10, 20)
SHARES = (0.25, 0.5, 0.75)


@torch.no_grad()
def rank_of(lm, h, M, ids):
    """rank (1 = top) of the best form among ids, through the map M (None: the logit lens), for states h [n, d]"""
    x = lm.model.unembed(h @ M.T if M is not None else h).float()
    v = x[:, ids].max(1).values
    return ((x > v[:, None]).sum(1) + 1).cpu().numpy()


def main():
    ap = parser(__doc__, "lens_onset")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--crop-set", type=int, default=1)
    args = ap.parse_args()
    require_data(args.data_dir, "coco")
    run = Run(args, "lens_onset")
    lm = Session(args.model)
    run.write_config(lm)
    crops = stimuli.load_crops(lm, args.data_dir / "coco2017", args.crop_set)
    crops.save(run)
    words, bank, sid = crops.words, crops.tiles, crops.ids
    t0 = time.time()
    rng = np.random.default_rng(29)
    X = [words[int(rng.integers(len(words)))] for _ in range(args.n)]
    O = [str(rng.choice([w for w in words if w != x])) for x in X]
    imgs = [stimuli.canvas(1, 1, {1: bank[x][i % len(bank[x])]}) for i, x in enumerate(X)]
    txt = [lm.chat_media([INTRO, "IMAGE"])] * args.n
    LF = lm.all_layers()
    layers = [L for L in LF if L >= 0]
    maps = {"J": {L: lm.lens.jacobians[L].to(lm.device) for L in layers if L in lm.lens.jacobians}, "logit": {L: None for L in layers}}
    if lm.lens_jpp is not None:
        maps["J++"] = {L: lm.lens_jpp.jacobians[L].to(lm.device) for L in layers if L in lm.lens_jpp.jacobians}
    rows = []
    for j in range(args.n):
        _, R = lm.forward(txt[j : j + 1], imgs[j : j + 1], layers, where=lambda ids: [k for k, t in enumerate(ids) if t == lm.image_token_id])
        for L in layers:
            h = R[L][0].to(lm.device)  # [tokens, d]
            pooled = h.mean(0, keepdim=True)
            for name, mm in maps.items():
                if L not in mm:
                    continue
                r_pool = int(rank_of(lm, pooled, mm[L], sid[X[j]])[0])
                n_pool = int(rank_of(lm, pooled, mm[L], sid[O[j]])[0])
                r_tok = rank_of(lm, h, mm[L], sid[X[j]])
                n_tok = rank_of(lm, h, mm[L], sid[O[j]])
                rows.append(
                    dict(
                        scene=j,
                        X=X[j],
                        null=O[j],
                        layer=L,
                        lens=name,
                        rank_pooled=r_pool,
                        null_pooled=n_pool,
                        best_token_rank=int(r_tok.min()),
                        null_best_token_rank=int(n_tok.min()),
                        share_tokens_top10=float((r_tok <= 10).mean()),
                    )
                )
        if j % 16 == 0:
            print(f"scene {j}  [{time.time() - t0:.0f}s]")
    D = pd.DataFrame(rows)
    run.save_csv(D, "lens_onset.csv")
    n_layers = lm.final + 1
    out = dict(n=args.n, n_layers=n_layers, layers=layers, lenses=list(maps), curves={}, onsets={})
    for name in maps:
        d = D[D.lens == name]
        out["curves"][name] = {}
        out["onsets"][name] = {}
        for col, lab in (("rank_pooled", "pooled"), ("best_token_rank", "best token")):
            for k in TOPS:
                c = d.groupby("layer")[col].apply(lambda v, k=k: float((v <= k).mean())).reindex(layers).fillna(0)
                cn = d.groupby("layer")[col.replace("rank", "null") if col == "rank_pooled" else "null_best_token_rank"].apply(lambda v, k=k: float((v <= k).mean())).reindex(layers).fillna(0)
                out["curves"][name][f"{lab} top{k}"] = c.tolist()
                out["curves"][name][f"{lab} top{k} null"] = cn.tolist()
                for s in SHARES:
                    on = next((L for L, v in zip(layers, c.values) if v >= s), None)
                    out["onsets"][name][f"{lab} top{k} share{s}"] = dict(layer=on, frac=(None if on is None else (on + 1) / n_layers))
    try:
        import json

        K = json.load(open(args.runs_dir / "run7k" / lm.tag / "summary.json"))
        out["knockout_last_single_reader_animal"] = max(K["questions"]["animal"]["readers"], default=None)
        out["knockout_last_single_reader_say"] = max(K["questions"]["say"]["readers"], default=None)
    except Exception as e:  # noqa: BLE001 - the knockout may not have run for this model
        out["knockout_note"] = f"no knockout summary read: {type(e).__name__}"
    for name in maps:
        print(f"{name}: pooled top-10 share by layer " + " ".join(f"{L}:{v:.2f}" for L, v in zip(layers, out["curves"][name]["pooled top10"])))
        print(f"{name}: onset pooled top10 half {out['onsets'][name]['pooled top10 share0.5']}, best-token top10 half {out['onsets'][name]['best token top10 share0.5']}")
    out["seconds"] = time.time() - t0
    run.save_json(out, "summary.json")


if __name__ == "__main__":
    main()
