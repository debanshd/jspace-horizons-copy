#!/usr/bin/env python
"""Why does Gemma-4-12B's lens fail scripts/check_lens.py? A diagnosis with no pass/fail rule: it measures, and the next step is
decided from what it shows (a new check is written before it runs, never a changed one after a result).

    uv run scripts/diagnose_lens.py --model google/gemma-4-E4B-it --run diagnose_lens        # the reference: a lens that passes
    uv run scripts/diagnose_lens.py --model google/gemma-4-12B-it --run diagnose_lens \\
        --extra-lens lamm-mit/gemma4-jacobian-lenses:gemma4-12b-it/paper/seed0.pt@c095b094c52dd7afeb1310bb97ea35dc5005e12f

check_lens found, on the 12B, that both its lens (hlampes) and an independently fitted one (lamm-mit, 1000 prompts) rarely
agree with the model's prediction, and that the plain logit lens at layer 46 agrees on 0.5% of positions (Gemma-4-E4B's at its
layer 39: 37%). Either the readout the lens library computes for this model is wrong, or the model's last layer really changes
its prediction that much. On check_lens's 48 prompts, per model:
1. target: the model's prediction as jlens computes it (the last layer's residual, the final norm, the unembedding, the
   softcap) against the model's own forward (Session.hf on the same token ids): top-1 agreement and the largest log-probability
   difference. A disagreement means the lens library reads this model wrongly, and every check_lens number is void.
2. every layer, not only the read layers: for the J-lens and the plain logit lens, the top-1 agreement with the model's
   prediction and the median rank of the model's top token in the readout (1 is the top), so a near miss counts.
3. per layer: the mean norm of the residual at the positions, and each decoder layer's layer_scalar (Gemma-4's per-layer
   output scale, where the model has one): a jump at the last layer would show where the prediction is made.
4. with --extra-lens: the same for the other lens at its source layers.
Outputs in runs/<run>/<model>/: diagnose.csv (per layer and readout), summary.json, config.json, run.log.
"""

import json

import numpy as np
import pandas as pd
import torch

from jspace.model import Session
from jspace.runs import Run, parser


def prompts(lm):
    """check_lens's prompts, as check_lens reads them"""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("check_lens", Path(__file__).with_name("check_lens.py"))
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    return [(lm.tokenizer.bos_token or "") + t for t in cl.prompts()]


def readout_stats(lens_logits, model_top):
    """per layer: (top-1 agreement count, ranks of the model's top token) over the positions"""
    out = {}
    for L, lg in lens_logits.items():
        top_score = lg.gather(1, model_top[:, None])
        ranks = (lg > top_score).sum(1) + 1
        out[L] = (int((lg.argmax(-1) == model_top).sum()), ranks.tolist())
    return out


def main():
    ap = parser(__doc__, "diagnose_lens")
    ap.add_argument("--extra-lens", help="repo:file@revision of another lens for this model")
    args = ap.parse_args()
    run = Run(args, "diagnose_lens")
    lm = Session(args.model)
    run.write_config(lm)
    texts = prompts(lm)
    lenses = {"model's": lm.lens}
    if args.extra_lens:
        from jlens import JacobianLens

        repo, rest = args.extra_lens.split(":", 1)
        file, rev = rest.split("@")
        lenses[args.extra_lens] = JacobianLens.from_pretrained(repo, filename=file, revision=rev)
    n = lm.final + 1
    acc = {}  # (lens, kind, L) -> [hits, ranks]
    norms = {L: [] for L in range(n)}
    target = dict(positions=0, agree=0, max_dlogp=0.0, readout_positions=0)
    for t in texts:
        # 1. the model's own forward on the token ids jlens reads, with the residual norm of every layer on the way
        ids = lm.model.encode(t)
        hooks = [lm.register(L, (lambda L: lambda mod, inp, out: norms[L].append(float((out[0] if isinstance(out, tuple) else out)[0].float().norm(dim=-1).mean())))(L)) for L in range(n)]
        try:
            with torch.no_grad():
                hf_logits = lm.hf(input_ids=ids, use_cache=False).logits[0].float().cpu()
        except Exception as e:  # noqa: BLE001 - the rest still measures; the error is the finding
            hf_logits = None
            target["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            for h in hooks:
                h.remove()
        for name, lens in lenses.items():
            src = list(lens.source_layers)
            for kind, use in (("jlens", True), ("logit", False)):
                if kind == "logit" and name != "model's":
                    continue  # the logit lens does not depend on the lens
                with torch.no_grad():
                    lens_logits, model_logits, jl_ids = lens.apply(lm.model, t, layers=src, use_jacobian=use)
                assert torch.equal(jl_ids.cpu(), ids.cpu()), "jlens read other token ids"
                top = model_logits.argmax(-1)
                if name == "model's" and kind == "jlens":
                    target["readout_positions"] += int(top.numel())
                if name == "model's" and kind == "jlens" and hf_logits is not None:
                    target["positions"] += int(top.numel())  # compared with the model's own forward
                    target["agree"] += int((hf_logits.argmax(-1) == top).sum())
                    d = (torch.log_softmax(hf_logits, -1) - torch.log_softmax(model_logits.float(), -1)).abs().max()
                    target["max_dlogp"] = max(target["max_dlogp"], float(d))
                for L, (hit, ranks) in readout_stats(lens_logits, top).items():
                    a = acc.setdefault((name, kind, L), [0, []])
                    a[0] += hit
                    a[1] += ranks
    rows = [
        dict(lens=name, readout=kind, layer=L, depth=L / n, top1=h / target["readout_positions"], median_rank=float(np.median(r)), mean_norm=float(np.mean(norms[L])))
        for (name, kind, L), (h, r) in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    ]
    R = pd.DataFrame(rows)
    run.save_csv(R, "diagnose.csv")
    scal = [float(lm.model.layers[L].layer_scalar.float().flatten()[0]) if hasattr(lm.model.layers[L], "layer_scalar") else None for L in range(n)]
    out = dict(
        target=dict(target, agree_share=target["agree"] / max(target["positions"], 1)),
        layer_scalar=scal,
        mean_norm={L: float(np.mean(v)) for L, v in norms.items()},
        lenses={name: dict(n_prompts=int(lens.n_prompts), source_layers=list(lens.source_layers)) for name, lens in lenses.items()},
    )
    run.save_json(out, "summary.json")
    print(f"target: jlens's model prediction = the model's own forward on {out['target']['agree_share']:.3f} of {target['positions']} positions (max |d log p| {target['max_dlogp']:.3f})")
    print(f"layer_scalar {json.dumps([None if s is None else round(s, 4) for s in scal])}")
    for _, r in R.iterrows():
        print(f"{r['lens'][:28]:28s} {r['readout']:5s} layer {int(r['layer']):2d}: top-1 {r['top1']:.3f}  median rank {r['median_rank']:8.0f}  residual norm {r['mean_norm']:9.1f}")


if __name__ == "__main__":
    main()
