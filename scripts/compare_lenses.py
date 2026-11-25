#!/usr/bin/env python
"""Compare two J-lenses of the same model, layer by layer: checks 1 and 2 of config.LENS_CHECKS (the rules are written there).

    uv run scripts/compare_lenses.py --model Qwen/Qwen3.5-2B --a pinned --b runs/lens_repro_fit --run lens_repro        # check 1
    uv run scripts/compare_lenses.py --model Qwen/Qwen2.5-Omni-3B --a runs/fit_lens_h0 --b runs/fit_lens_ext1000b --run lens_halves  # 2

A lens is "pinned" (config.MODELS's lens of the model, its sha256 checked) or a run folder of scripts/fit_lens.py (its
<model>/lens.pt and meta.json). Per layer both lenses have: the cosine of the two maps flattened, and of their departures from
the identity (the logit lens's map), reported beside it. passes: the first cosine is at least LENS_CHECKS's min_cosine at every
layer from 0.3 of the depth. No GPU and no model are needed.

Outputs in runs/<run>/<model>/: compare.csv (per layer), compare.json (the two lenses, with sha256s and prompt ranges, the
checked layers, the lowest cosine, passes), run.log.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from jspace import config, lenscheck
from jspace.runs import Run, load_env


def load(model, source):
    """(JacobianLens, a description with its sha256 and, for a fitted lens, its meta.json's prompt range)"""
    from jlens import JacobianLens

    if source == "pinned":
        from huggingface_hub import hf_hub_download

        spec = config.MODELS[config.REPRO_PINNED.get(model, model)]  # a base model refitted for check 1: its instruct model's pin
        repo, rev = spec.lens_source()
        path = Path(hf_hub_download(repo, spec.lens_file, revision=rev))
        got = lenscheck.sha256(path)
        if got != spec.lens_sha256:
            raise SystemExit(f"{repo}/{spec.lens_file}: sha256 {got}, pinned {spec.lens_sha256}")
        return JacobianLens.load(str(path)), dict(source="pinned", repo=repo, revision=rev, file=spec.lens_file, sha256=got)
    d = Path(source) / model.split("/")[-1]
    if not (d / "lens.pt").exists():
        raise SystemExit(f"{d / 'lens.pt'} not found")
    meta = json.loads((d / "meta.json").read_text())
    if meta["model"] != model:
        raise SystemExit(f"{d}: a lens of {meta['model']}, not {model}")
    got = lenscheck.sha256(d / "lens.pt")
    if got != meta["lens_sha256"]:
        raise SystemExit(f"{d / 'lens.pt'}: sha256 {got}, its meta.json says {meta['lens_sha256']}")
    desc = dict(source=str(source), sha256=got, n_prompts=meta["n_prompts"], n_requested=meta["n_requested"], start=meta.get("start", 0), shard=meta.get("shard"))
    return JacobianLens.load(str(d / "lens.pt")), desc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--a", required=True, help='"pinned" or a scripts/fit_lens.py run folder')
    ap.add_argument("--b", required=True, help='"pinned" or a scripts/fit_lens.py run folder')
    ap.add_argument("--run", required=True)
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    args = ap.parse_args()
    load_env()
    run = Run(argparse.Namespace(model=args.model, run=args.run, runs_dir=args.runs_dir), "compare_lenses")
    la, da = load(args.model, args.a)
    lb, db = load(args.model, args.b)
    n_layers = max(max(la.source_layers), max(lb.source_layers)) + 2  # the source layers are every block before the final one
    cos = lenscheck.cosines(la.jacobians, lb.jacobians)
    passes, layers, low = lenscheck.agree(cos, n_layers)
    R = pd.DataFrame([dict(layer=L, depth=L / n_layers, cosine=c, cosine_departure=c2, checked=L in layers) for L, (c, c2) in cos.items()])
    run.save_csv(R, "compare.csv")
    for _, r in R.iterrows():
        print(f"layer {int(r['layer']):2d} ({r['depth']:.2f}){' *' if r['checked'] else '  '}: cosine {r['cosine']:.4f}, of the departures from the identity {r['cosine_departure']:.4f}")
    out = dict(model=args.model, a=da, b=db, n_layers=n_layers, min_cosine=config.LENS_CHECKS["min_cosine"], checked_layers=layers, lowest=low, passes=passes)
    out["cosine"] = {int(L): c for L, (c, _) in cos.items()}
    out["cosine_departure"] = {int(L): c2 for L, (_, c2) in cos.items()}
    run.save_json(out, "compare.json")
    print(f"lowest cosine over the checked layers (*) {low:.4f}, against {config.LENS_CHECKS['min_cosine']}: {'passes' if passes else 'FAILS'}")


if __name__ == "__main__":
    main()
