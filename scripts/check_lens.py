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

Retired 25 Nov, having judged nothing (void: runs/check_lens_calibration_ranks); lenses fitted here are validated by
config.LENS_CHECKS instead (scripts/validate_lens.py). Kept as the record: The Qwen rule (the second version, written 25 Nov before any lens is judged by it; config.QWEN_CALIBRATION). The rule above
failed three of the four Neuronpedia Qwen lenses the original results read, which rarely make the model's top token their top
one before the last layers. So for a Qwen lens:
- the measure (--ranks): per read layer, the median over positions of log10 of the model's top token's rank in the readout (1
  is the top), for the J-lens and the plain logit lens. Lower is better.
- the bar: the calibration run (--calibration, scripts/lens_calibration.sh) holds the four Neuronpedia Qwen lenses' ranks. At a
  depth, the bar is the highest (worst) J-lens rank of the four at their read layer nearest that depth, so each of them meets it.
- informative: the bar must reject each of the four models' own logit lens, i.e. each logit lens is above the bar at some read
  layer from half depth. Otherwise a lens with no Jacobian at all would pass, the rule says nothing, and passes is false.
- passes: informative, and at every read layer from half depth on, the lens's J-lens rank is at or below the bar.
- only lenses checked for the first time: a lens in config.SEEN_LENSES (seen before this rule was written) is refused.
beats_logit_lens and the top-token agreements are still reported; with --calibration they decide nothing.
Outputs in runs/<run>/<model>/: check_lens.csv (per layer and lens), summary.json, config.json, run.log.
"""

import json

import numpy as np
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


def agreement(lm, lens, layers, texts, ranks=False):
    """per layer: the share of positions where the lens's top token is the model's (J-lens, and the logit lens); with ranks,
    also the median over positions of log10 of the model's top token's rank in the readout"""
    hit = {(L, kind): 0 for L in layers for kind in ("jlens", "logit")}
    rk = {(L, kind): [] for L in layers for kind in ("jlens", "logit")}
    n = 0
    for t in texts:
        for kind, use in (("jlens", True), ("logit", False)):
            lens_logits, model_logits, _ = lens.apply(lm.model, t, layers=layers, use_jacobian=use)
            top = model_logits.argmax(-1)
            for L in layers:
                hit[(L, kind)] += int((lens_logits[L].argmax(-1) == top).sum())
                if ranks:
                    lg = lens_logits[L].float()
                    rk[(L, kind)] += ((lg > lg.gather(-1, top[:, None])).sum(-1) + 1).tolist()
        n += int(top.numel())
    got = {k: v / n for k, v in hit.items()}
    if ranks:
        return got, n, {k: float(np.median(np.log10(v))) for k, v in rk.items()}
    return got, n


def nearest(by_layer, n_layers, d):
    """a summary's per-layer value at its read layer nearest depth d"""
    by_depth = {int(L) / n_layers: v for L, v in by_layer.items()}
    return by_depth[min(by_depth, key=lambda x: abs(x - d))]


def calibration_bar(summaries, depths):
    """per depth: the highest (worst) J-lens log10 rank among the calibration summaries, each at its read layer nearest that depth"""
    return [max(nearest(s["rank_jlens"], s["n_layers"], d) for s in summaries) for d in depths]


def informative(summaries):
    """per calibration model: whether the bar rejects its plain logit lens (above the bar at some read layer from half depth)"""
    out = {}
    for m, s in summaries.items():
        late = [int(L) for L in s["rank_logit"] if int(L) / s["n_layers"] >= 0.5]
        bar = calibration_bar(list(summaries.values()), [L / s["n_layers"] for L in late])
        out[m] = bool(any(s["rank_logit"][str(L)] > b for L, b in zip(late, bar)))
    return out


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
        if "rank_jlens" not in s:
            raise SystemExit(f"{f} has no ranks: scripts/lens_calibration.sh runs check_lens with --ranks")
        out[m] = s
    return out


def main():
    ap = parser(__doc__, "check_lens")
    ap.add_argument("--reference", help="a model whose check_lens summary in this run sets the bar (near_reference)")
    ap.add_argument("--lens", help="check this alternative lens of the model (config.ALT_LENSES, e.g. Qwen3.5-9B's jlens), not its pinned one")
    ap.add_argument("--ranks", action="store_true", help="also measure the model's top token's rank in each readout (the Qwen rule's measure)")
    ap.add_argument("--calibration", help="a run holding the Qwen calibration lenses' ranks: judge by the Qwen rule (implies --ranks)")
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
        sha = config.lens_for(args.model, args.lens)[3]
        if sha in config.SEEN_LENSES:
            raise SystemExit(f"{config.SEEN_LENSES[sha]}: its numbers were seen before the Qwen rule was written, so the rule does not judge it")
        cal = load_calibration(args.runs_dir, args.calibration)
        args.ranks = True
    run = Run(args, "check_lens")
    lm = Session(args.model, lens=args.lens)
    run.write_config(lm)
    # the Session's lens model adds no BOS (its chat templates carry one); plain text gets the tokenizer's own, as in training
    texts = [(lm.tokenizer.bos_token or "") + t for t in prompts()]
    n = lm.final + 1
    with torch.no_grad():
        got, positions, *rk = agreement(lm, lm.lens, lm.layers, texts, ranks=args.ranks)
    rows = [dict(lens="model's", layer=L, depth=L / n, jlens=got[(L, "jlens")], logit=got[(L, "logit")]) for L in lm.layers]
    if args.ranks:  # only then, so a check_lens without it writes what it always wrote
        for r in rows:
            r.update(rank_jlens=rk[0][(r["layer"], "jlens")], rank_logit=rk[0][(r["layer"], "logit")])
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
            got2, _, *_ = agreement(lm, other, shared, texts)
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
    if args.ranks:
        out["rank_jlens"] = {int(L): float(v) for L, v in zip(mine["layer"], mine["rank_jlens"])}
        out["rank_logit"] = {int(L): float(v) for L, v in zip(mine["layer"], mine["rank_logit"])}
    if cal is not None:
        bar = calibration_bar(list(cal.values()), list(late["depth"]))
        inf = informative(cal)
        out["calibration"] = dict(run=args.calibration, models=list(cal), bar={int(L): b for L, b in zip(late["layer"], bar)}, rejects_logit_lens=inf)
        out["informative"] = bool(all(inf.values()))
        out["meets_calibration"] = bool(all(v <= b for v, b in zip(late["rank_jlens"], bar)))
        out["rule"] = "qwen2"
        out["passes"] = bool(out["informative"] and out["meets_calibration"])
    run.save_json(out, "summary.json")
    for _, r in R.iterrows():
        print(f"{r['lens'][:40]:40s} layer {int(r['layer']):2d} ({r['depth']:.2f}): J-lens {r['jlens']:.3f}  logit lens {r['logit']:.3f}")
    print(f"deepest {out['deepest']:.3f}; beats the logit lens from half depth: {out['beats_logit_lens']}" + (f"; reference {ref['deepest']:.3f}; passes: {out['passes']}" if ref else ""))
    if cal is not None:
        print(f"Qwen rule: J-lens log10 rank {out['rank_jlens']}; bar {out['calibration']['bar']}; the bar rejects each logit lens: {inf}; passes: {out['passes']}")


if __name__ == "__main__":
    main()
