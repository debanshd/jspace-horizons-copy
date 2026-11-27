"""Cosine similarity, layer by layer, between the edited lenses, the 100% source and the fitted 50% lens (the target).

Each layer's 4096 x 4096 matrix is flattened, so the cosine compares whole matrices; the readout normalises the transported residual,
so a lens's overall scale does not matter and the cosine is the natural comparison. Exploratory: the rule set no verdict on it. Run
only after the freeze (frozen.json), since it reads the target.

    python jpp_wsp/cosine.py --lens name=path [...] --out jpp_wsp/cosine.json
"""

import argparse
import itertools
import json
from pathlib import Path

import torch

READ_LAYERS = [4, 8, 12, 16, 20, 24, 28]  # the seven layers the J++ evals read on a 32-layer model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    # memory-mapped, so five 2 GB lenses are read one layer at a time
    lenses = {n: torch.load(p, map_location="cpu", weights_only=True, mmap=True) for n, p in (s.split("=", 1) for s in args.lens)}
    layers = next(iter(lenses.values()))["source_layers"]
    pairs = list(itertools.combinations(lenses, 2))
    cos = {f"{a} | {b}": {} for a, b in pairs}
    for L in layers:
        flat = {n: c["parameters"]["jacobians"][L].double().flatten() for n, c in lenses.items()}
        for a, b in pairs:
            cos[f"{a} | {b}"][L] = float(torch.nn.functional.cosine_similarity(flat[a], flat[b], dim=0))
        del flat
    summary = {
        pair: {
            "mean_all_layers": sum(v.values()) / len(v),
            "mean_read_layers": sum(v[L] for L in READ_LAYERS) / len(READ_LAYERS),
            "read_layers": {L: round(v[L], 4) for L in READ_LAYERS},
        }
        for pair, v in cos.items()
    }
    args.out.write_text(json.dumps({"summary": summary, "per_layer": cos}, indent=1))
    for pair, s in summary.items():
        print(f"{pair:45s} all {s['mean_all_layers']:.3f}  read {s['mean_read_layers']:.3f}  {s['read_layers']}")


if __name__ == "__main__":
    main()
