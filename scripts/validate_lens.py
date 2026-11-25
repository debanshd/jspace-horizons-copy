#!/usr/bin/env python
"""Whether a lens fitted here may be used: config.LENS_CHECKS's three checks (written there before any of them ran), read from
the runs that measured them. It measures nothing itself and needs no GPU.

    uv run scripts/validate_lens.py --model Qwen/Qwen2.5-Omni-3B --lens runs/fit_lens_n500

1. reproduce: runs/lens_repro/Qwen3.5-2B/compare.json (scripts/compare_lenses.py: this recipe's 283-prompt refit of Qwen3.5-2B
   against Neuronpedia's lens of it) passes.
2. stable: runs/lens_halves/<model>/compare.json passes, and its two lenses are this lens's halves: prompts 0-249 (the
   250-prompt fit) and 250-499, the second of which the lens's merge contains.
3. reads concepts: runs/lens_readout/<model>/summary.json, of this lens (its sha256): its margin over its own logit lens against
   the lowest margin among runs/lens_readout_np/<model>/summary.json for config.READOUT_CALIBRATION (Neuronpedia's Qwen3.5
   J-lenses, each the lens config.MODELS pins), and whether that lowest margin is above 0. Since the change of 25 Nov
   (config.LENS_CHECKS, readout_decides; approved by the project owner after check 3 came out void) it is reported as a caveat
   and decides nothing: the video's lenses fail it too. passes: checks 1 and 2.
A missing input stops it with an error. Only Qwen lenses: the calibration lenses are Qwen's; a Gemma lens needs its own
calibration set, written before it is validated.

Outputs in runs/lens_validation/<model>/: summary.json (each check with its numbers, and passes), run.log.
"""

import argparse
import json
from pathlib import Path

from jspace import config, lenscheck
from jspace.runs import Run


def read(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found: run the step that writes it first (scripts/fit_lenses.sh)")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True, type=Path, help="the scripts/fit_lens.py run folder of the lens (its <model>/lens.pt)")
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    args = ap.parse_args()
    if not args.model.startswith("Qwen/"):
        raise SystemExit("only Qwen lenses: the readout calibration (config.READOUT_CALIBRATION) is Neuronpedia's Qwen lenses")
    run = Run(argparse.Namespace(model=args.model, run="lens_validation", runs_dir=args.runs_dir), "validate_lens")
    tag, R = args.model.split("/")[-1], args.runs_dir
    meta = read(args.lens / tag / "meta.json")
    sha = lenscheck.sha256(args.lens / tag / "lens.pt")
    if sha != meta["lens_sha256"]:
        raise SystemExit(f"{args.lens / tag / 'lens.pt'}: sha256 {sha}, its meta.json says {meta['lens_sha256']}")
    lo, hi = config.LENS_CHECKS["halves"]

    repro = read(R / "lens_repro" / config.LENS_CHECKS["reproduce"].split("/")[-1] / "compare.json")
    if repro["a"]["source"] != "pinned" or repro["b"].get("n_requested") != config.LENS_CHECKS["reproduce_n"]:
        raise SystemExit("runs/lens_repro is not the reproduction check: Neuronpedia's pinned lens against a refit of reproduce_n prompts")
    check1 = dict(passes=repro["passes"], lowest=repro["lowest"], layers=repro["checked_layers"])

    halves = read(R / "lens_halves" / tag / "compare.json")
    a, b = halves["a"], halves["b"]
    is_halves = (a.get("n_requested"), a.get("start"), a.get("shard"), b.get("n_requested"), b.get("start"), b.get("shard")) == (lo, 0, None, hi, lo, None)
    in_lens = meta.get("n_requested") == hi and b["sha256"] in [p["lens_sha256"] for p in meta.get("merged", [])]
    if not (is_halves and in_lens):
        raise SystemExit(f"runs/lens_halves/{tag} is not this lens's halves (prompts 0-{lo - 1} and {lo}-{hi - 1}, the second merged into it)")
    check2 = dict(passes=halves["passes"], lowest=halves["lowest"], layers=halves["checked_layers"])

    mine = read(R / "lens_readout" / tag / "summary.json")
    if mine["lens"]["sha256"] != sha:
        raise SystemExit(f"runs/lens_readout/{tag} read another lens ({mine['lens']['sha256'][:12]}), not {sha[:12]}")
    cal = {}
    for m in config.READOUT_CALIBRATION:
        s = read(R / "lens_readout_np" / m.split("/")[-1] / "summary.json")
        if s["lens"]["source"] != "pinned" or s["lens"]["sha256"] != config.MODELS[m].lens_sha256:
            raise SystemExit(f"runs/lens_readout_np/{m.split('/')[-1]} is not {m}'s pinned lens")
        cal[m] = lenscheck.margin(s)
    bar, informative = lenscheck.readout_bar(cal)
    k = config.LENS_CHECKS["k"]
    mine_margin = lenscheck.margin(mine)
    check3 = dict(
        passes=bool(informative and mine_margin >= bar), informative=informative, margin=mine_margin, bar=bar, calibration_margins=cal, score=mine["score"], logit_lens=mine[f"logit_pass@{k}"]["mean"]
    )

    out = dict(model=args.model, lens=dict(run=str(args.lens), sha256=sha, n_prompts=meta["n_prompts"]), reproduce=check1, stable=check2, reads_concepts=check3)
    check3["decides"] = config.LENS_CHECKS["readout_decides"]
    out["passes"] = bool(check1["passes"] and check2["passes"] and (check3["passes"] or not check3["decides"]))
    run.save_json(out, "summary.json")
    print(f"1. reproduce: lowest cosine {check1['lowest']:.4f}: {check1['passes']}")
    print(f"2. stable: lowest cosine between the halves {check2['lowest']:.4f}: {check2['passes']}")
    print(
        f"3. reads concepts: pass@{k} {check3['score']:.3f}, its logit lens {check3['logit_lens']:.3f}, margin {mine_margin:+.3f} against the bar {bar:+.3f} {cal}"
        f"{'' if informative else ' (VOID: a calibration lens does not beat its logit lens)'}: {check3['passes']}"
        f"{'' if check3['decides'] else ' (a caveat; decides nothing, config.LENS_CHECKS)'}"
    )
    print(f"{args.model}: {'passes' if out['passes'] else 'FAILS'}")


if __name__ == "__main__":
    main()
