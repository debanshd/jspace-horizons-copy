#!/usr/bin/env python
"""Check a model's J-lens on the model itself before any experiment reads it, against a reference model whose lens the runs
already use. Written for Gemma-4-12B-it, whose lens (hlampes/jacobian-lens) is not Neuronpedia's and has no card.

    uv run scripts/check_lens.py --model google/gemma-4-E4B-it --run check_lens                        # the reference first
    uv run scripts/check_lens.py --model google/gemma-4-12B-it --run check_lens --reference google/gemma-4-E4B-it \\
        --extra-lens lamm-mit/gemma4-jacobian-lenses:gemma4-12b-it/paper/seed0.pt@c095b094c52dd7afeb1310bb97ea35dc5005e12f

What it measures. The model reads prompts() (48 plain sentences about the 16 objects of the experiments, no chat template). At
every position and every read layer of the model (Session.layers), the J-lens readout (the residual carried to the final layer
by the lens's Jacobian, then unembedded) and the plain logit lens (the residual unembedded as it is) are compared with the
model's own next-token prediction: the share of positions whose top token agrees. A lens fitted to this model should beat the
logit lens and approach the model's prediction with depth, as Neuronpedia's lenses do. --extra-lens reads another lens file
for the same model at the read layers it shares (reported, not judged): two lenses of the same model should agree in shape.

The rule (written before the run), with --reference (that model's own check_lens summary must exist in the same run):
- beats_logit_lens: at every read layer from half depth on, the J-lens agrees with the model at least as often as the logit lens.
- near_reference: at the deepest read layer, the J-lens agreement is at least the reference model's at its deepest read layer
  minus 0.10.
- passes: both. scripts/audio_sweep.sh runs a model with a lens from outside Neuronpedia only when it passes.

The Qwen rule (written 25 Nov, before any lens is judged by it; config.QWEN_CALIBRATION), with --calibration <run> in place of
--reference: the rule above failed three of the four Neuronpedia Qwen lenses the original results read, so for a Qwen lens:
- meets_calibration: at every read layer from half depth on, the J-lens agreement is at least the lowest agreement of the four
  calibration lenses (their check_lens summaries in runs/<calibration run>/) at their read layer nearest that depth.
- passes: meets_calibration. beats_logit_lens is still reported, and decides nothing.
Outputs in runs/<run>/<model>/: check_lens.csv (per layer and lens), summary.json, config.json, run.log.
"""

import json

import pandas as pd
import torch

from jspace import config, stimuli
from jspace.model import Session
from jspace.runs import Run, parser

TEMPLATES = ["The {w} in the old photograph was", "She pointed at the {w} and said that it", "Everyone knows that a {w} is usually"]
MARGIN = 0.10


def prompts():
    """PROMPTS: every template with every object of the experiments"""
    return [t.format(w=w) for w in stimuli.VOCAB for t in TEMPLATES]


def agreement(lm, lens, layers, texts):
    """per layer: the share of positions where the lens's top token is the model's (J-lens, and the logit lens)"""
    hit = {(L, kind): 0 for L in layers for kind in ("jlens", "logit")}
    n = 0
    for t in texts:
        for kind, use in (("jlens", True), ("logit", False)):
            lens_logits, model_logits, _ = lens.apply(lm.model, t, layers=layers, use_jacobian=use)
            top = model_logits.argmax(-1)
            for L in layers:
                hit[(L, kind)] += int((lens_logits[L].argmax(-1) == top).sum())
        n += int(top.numel())
    return {k: v / n for k, v in hit.items()}, n


def calibration_bar(summaries, depths):
    """per depth: the lowest J-lens agreement among the calibration summaries, each read at its read layer nearest that depth"""
    bar = []
    for d in depths:
        vals = []
        for s in summaries:
            by_depth = {int(L) / s["n_layers"]: v for L, v in s["jlens"].items()}
            vals.append(by_depth[min(by_depth, key=lambda x: abs(x - d))])
        bar.append(min(vals))
    return bar


def load_calibration(runs_dir, run):
    """the calibration lenses' check_lens summaries: each must exist and be their pinned (Neuronpedia) lens"""
    out = {}
    for m in config.QWEN_CALIBRATION:
        f = runs_dir / run / m.split("/")[-1] / "summary.json"
        if not f.exists():
            raise SystemExit(f"{f} not found: run scripts/lens_calibration.sh first")
        s = json.loads(f.read_text())
        if s["lens"].get("alt") or s["lens"]["file"] != config.MODELS[m].lens_file:
            raise SystemExit(f"{f} is not {m}'s pinned lens")
        out[m] = s
    return out


def main():
    ap = parser(__doc__, "check_lens")
    ap.add_argument("--reference", help="a model whose check_lens summary in this run sets the bar (near_reference)")
    ap.add_argument("--lens", help="check this alternative lens of the model (config.ALT_LENSES, e.g. Qwen3.5-9B's jlens), not its pinned one")
    ap.add_argument("--calibration", help="a run holding the Qwen calibration lenses' summaries: judge by the Qwen rule")
    ap.add_argument("--extra-lens", help="repo:file@revision of another lens for this model, read at the layers it shares")
    args = ap.parse_args()
    ref = None
    if args.reference:
        f = args.runs_dir / args.run / args.reference.split("/")[-1] / "summary.json"
        if not f.exists():
            raise SystemExit(f"{f} not found: run check_lens for the reference {args.reference} first")
        ref = json.loads(f.read_text())
    cal = None
    if args.calibration:
        if not args.model.startswith("Qwen/"):
            raise SystemExit("the Qwen rule (--calibration) judges Qwen lenses only")
        cal = load_calibration(args.runs_dir, args.calibration)
    run = Run(args, "check_lens")
    lm = Session(args.model, lens=args.lens)
    run.write_config(lm)
    # the Session's lens model adds no BOS (its chat templates carry one); plain text gets the tokenizer's own, as in training
    texts = [(lm.tokenizer.bos_token or "") + t for t in prompts()]
    n = lm.final + 1
    with torch.no_grad():
        got, positions = agreement(lm, lm.lens, lm.layers, texts)
    rows = [dict(lens="model's", layer=L, depth=L / n, jlens=got[(L, "jlens")], logit=got[(L, "logit")]) for L in lm.layers]
    extra = None
    if args.extra_lens:
        repo, rest = args.extra_lens.split(":", 1)
        file, rev = rest.split("@")
        from jlens import JacobianLens  # here, so the prompts and the rule import without the lens library

        other = JacobianLens.from_pretrained(repo, filename=file, revision=rev)
        shared = [L for L in lm.layers if L in other.source_layers]
        if not shared:
            raise SystemExit(f"{args.extra_lens} shares no read layer with {lm.layers}")
        with torch.no_grad():
            got2, _ = agreement(lm, other, shared, texts)
        rows += [dict(lens=args.extra_lens, layer=L, depth=L / n, jlens=got2[(L, "jlens")], logit=got2[(L, "logit")]) for L in shared]
        extra = dict(lens=args.extra_lens, n_prompts=int(other.n_prompts), layers=shared, jlens={L: got2[(L, "jlens")] for L in shared}, model_lens={L: got[(L, "jlens")] for L in shared})
    R = pd.DataFrame(rows)
    run.save_csv(R, "check_lens.csv")
    repo, rev, file = lm.lens_source["repo"], lm.lens_source["revision"], lm.lens_source["file"]  # the lens the session read
    mine = R[R["lens"] == "model's"]
    late = mine[mine["depth"] >= 0.5]
    out = dict(
        lens=dict(repo=repo, revision=rev, file=file, alt=args.lens, n_prompts=int(lm.lens.n_prompts), source_layers=list(lm.lens.source_layers)),
        n_layers=n,
        read_layers=lm.layers,
        positions=positions,
        prompts=len(texts),
        jlens={int(L): float(v) for L, v in zip(mine["layer"], mine["jlens"])},
        logit={int(L): float(v) for L, v in zip(mine["layer"], mine["logit"])},
        deepest=float(mine["jlens"].iloc[-1]),
        beats_logit_lens=bool((late["jlens"] >= late["logit"]).all()),
        extra=extra,
    )
    if ref is not None:
        out["reference"] = dict(model=args.reference, deepest=ref["deepest"])
        out["near_reference"] = bool(out["deepest"] >= ref["deepest"] - MARGIN)
        out["passes"] = bool(out["beats_logit_lens"] and out["near_reference"])
    if cal is not None:
        bar = calibration_bar(list(cal.values()), list(late["depth"]))
        out["calibration"] = dict(run=args.calibration, models=list(cal), bar={int(L): b for L, b in zip(late["layer"], bar)})
        out["meets_calibration"] = bool(all(v >= b for v, b in zip(late["jlens"], bar)))
        out["rule"] = "qwen"
        out["passes"] = out["meets_calibration"]
    run.save_json(out, "summary.json")
    for _, r in R.iterrows():
        print(f"{r['lens'][:40]:40s} layer {int(r['layer']):2d} ({r['depth']:.2f}): J-lens {r['jlens']:.3f}  logit lens {r['logit']:.3f}")
    print(f"deepest {out['deepest']:.3f}; beats the logit lens from half depth: {out['beats_logit_lens']}" + (f"; reference {ref['deepest']:.3f}; passes: {out['passes']}" if ref else ""))
    if cal is not None:
        print(f"Qwen rule: bar by layer {out['calibration']['bar']}; passes: {out['passes']}")


if __name__ == "__main__":
    main()
