#!/usr/bin/env python
"""The concept readout of a J-lens on its model: check 3 of config.LENS_CHECKS (the rule is written there), the lens-quality
evaluation of the J-lens paper (Gurnee et al. 2026) on five of its six prompt sets (src/jspace/lens_eval, NOTICE.md).

    uv run scripts/lens_readout.py --model Qwen/Qwen3.5-2B --lens pinned --run lens_readout_np         # a Neuronpedia lens
    uv run scripts/lens_readout.py --model Qwen/Qwen2.5-Omni-3B --lens runs/fit_lens_n1000 --run lens_readout   # a fitted one

Each item is a prompt whose answer needs a concept the prompt never names (multihop: "Brazil" for "The ocean on the coast of the
country where Carnival is most famously celebrated is the "). At one position (the last token; for poetry the last newline, the
end of the couplet's first line), at every layer the lens maps, the readout ranks the vocabulary; an intermediate's rank at a
layer is the better of the first tokens of "word" and " word", and its score is its best rank over the layers. pass@k for a set
is the mean, over items, of the share of its intermediates ranked at most k; the score is pass@10 averaged over the five sets.
The same for the plain logit lens (the identity map), reported beside it. Every item is scored, as for every lens alike (no
filter on whether the model answers it). The sixth set, order-ops, is left out: its intermediates need synonym sets the paper's
code does not ship.

The model is loaded in bf16 as scripts/fit_lens.py loads it (config.FIT_MODELS), or as Session loads it (config.MODELS), and
read by jlens as the fit read it. A lens is "pinned" (config.MODELS's, its sha256 checked) or a scripts/fit_lens.py run folder.

Outputs in runs/<run>/<model>/: readout.csv (per item and intermediate: best rank and layer, both lenses), summary.json (pass@1
and pass@10 per set and averaged, both lenses, and the lens with its sha256), run.log.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jspace import config, lenscheck
from jspace.runs import Run, load_env


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_model(name):
    """the model in bf16 on the GPU and its tokenizer: as fit_lens loads it if it is one of FIT_MODELS, else as Session does"""
    import transformers

    if name in config.FIT_MODELS:
        return script("fit_lens").load(name)
    rev = config.MODELS[name].revision
    cls = transformers.AutoModelForMultimodalLM if "gemma" in name.lower() else transformers.AutoModelForImageTextToText
    hf = cls.from_pretrained(name, revision=rev, dtype=torch.bfloat16).cuda().eval()
    return hf, transformers.AutoTokenizer.from_pretrained(name, revision=rev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True, help='"pinned" or a scripts/fit_lens.py run folder')
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    args = ap.parse_args()
    load_env()
    assert torch.cuda.is_available(), "the readout needs a CUDA GPU"
    import jlens

    run = Run(argparse.Namespace(model=args.model, run=args.run, runs_dir=args.runs_dir), "lens_readout")
    sets = lenscheck.eval_sets()
    lens, desc = script("compare_lenses").load(args.model, args.lens)
    hf, tok = load_model(args.model)
    lm = jlens.from_hf(hf, tok)
    layers = list(lens.source_layers)
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    newline = {i for i in range(len(tok)) if "\n" in tok.decode([i])} if "poetry" in sets else set()
    k = config.LENS_CHECKS["k"]
    rows = []
    with torch.no_grad():
        for set_name, items in sets.items():
            for it in items:
                off = lenscheck.readout_offset(encode(it["prompt"]), set_name, newline)
                cands = [lenscheck.word_ids(encode, w) for w in it["intermediates"]]
                best = {}
                for kind, use in (("jlens", True), ("logit", False)):
                    lens_logits, _, _ = lens.apply(lm, it["prompt"], layers=layers, positions=[off], use_jacobian=use)
                    for w, ids in zip(it["intermediates"], cands):
                        per_layer = []
                        for L in layers:
                            lg = lens_logits[L][0].float()
                            per_layer.append(min(int((lg > lg[i]).sum()) + 1 for i in ids))
                        j = int(np.argmin(per_layer))
                        best[(kind, w)] = (per_layer[j], layers[j])
                for w in it["intermediates"]:
                    rows.append(
                        dict(
                            set=set_name,
                            item=it["name"],
                            word=w,
                            jlens_rank=best[("jlens", w)][0],
                            jlens_layer=best[("jlens", w)][1],
                            logit_rank=best[("logit", w)][0],
                            logit_layer=best[("logit", w)][1],
                        )
                    )
    R = pd.DataFrame(rows)
    run.save_csv(R, "readout.csv")
    score = {}
    for kind in ("jlens", "logit"):
        for kk in (1, k):
            per_set = {s: lenscheck.pass_at_k([g[f"{kind}_rank"].tolist() for _, g in R[R["set"] == s].groupby("item", sort=False)], kk) for s in sets}
            score[f"{kind}_pass@{kk}"] = dict(per_set, mean=float(np.mean(list(per_set.values()))))
    out = dict(model=args.model, lens=desc, layers=layers, k=k, sets={s: len(v) for s, v in sets.items()}, **score)
    out["score"] = score[f"jlens_pass@{k}"]["mean"]
    run.save_json(out, "summary.json")
    for name, v in score.items():
        print(f"{name:14s} " + "  ".join(f"{s} {x:.3f}" for s, x in v.items()))
    print(f"score (J-lens pass@{k}, averaged over the sets): {out['score']:.3f}; the logit lens: {score[f'logit_pass@{k}']['mean']:.3f}")


if __name__ == "__main__":
    main()
