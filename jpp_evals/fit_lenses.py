"""Fit an R-lens and a J++ lens for Qwen2.5-Omni's thinker with the J++ post's recipe, in the post's harness.

The recipe is that of the post's Qwen3.6-27B lens and of Neuronpedia's Qwen3.5-4B R-lens and J++ lens (both fitted with this harness):
- a router of k-means clusters (E = 8 per layer) on the activations of the first 1000 WikiText-103 prompts (fit-router);
- expert Jacobian sums through the LRP backward pass ("rlens" rules) on the first 64 prompts, 128 tokens each, the first 16
  positions skipped (fit-shard), merged into the expert Jacobians (merge-experts);
- the R-lens: the pooled Jacobian of that fit, all experts together (merge-experts --pooled-lens-out);
- the J++ lens: the experts combined with weights fitted on the post's "fit:0" split of the readout items (fit-weights). A J++
  lens must then be scored on the held-out split only: run_all.py --fits does that.

Every source layer is fitted (0 to n - 2; the target is the last block), as in the omni J-lenses, so the lenses serve this
repository's experiments as well as the post's seven readout layers. A backward pass yields every source layer at once, so the extra
layers cost memory for their sums, not passes.

Qwen3-Omni-30B-A3B is left out: its MoE thinker has classes the LRP surgery has no verified rules for.

Per model, in <fits>/<model key>/: router.pt, artifacts/ (the shard sums), experts.pt, <prefix>_rlens_n<N>.pt,
<prefix>_jpp_n<N>.pt with its expert weights, correctness.csv, a log and the wall time of each step. A step whose output exists
is skipped.

Run from the root of a jpp_lens checkout, in its environment: uv run python <repo>/jpp_evals/fit_lenses.py --fits <dir> [keys ...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import model_by_key  # noqa: E402
from run_all import merged_correctness, record_environment, step  # noqa: E402

FITTABLE = ("qwen2.5-omni-3b", "qwen2.5-omni-7b")
PREFIX = {"qwen2.5-omni-3b": "omni3b", "qwen2.5-omni-7b": "omni7b"}


def lens_paths(fits_dir: Path, key: str, num_prompts: int) -> tuple[Path, Path]:
    """the R-lens and J++ lens files of a fit; their stems name the lenses in the eval tables"""
    d = fits_dir / key
    return d / f"{PREFIX[key]}_rlens_n{num_prompts}.pt", d / f"{PREFIX[key]}_jpp_n{num_prompts}.pt"


def fit_model(key: str, fits_dir: Path, num_prompts: int, router_prompts: int) -> None:
    m = model_by_key(key)
    d = fits_dir / key
    d.mkdir(parents=True, exist_ok=True)
    timings_path = d / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    hf = ["--hf-model-name", m.hf_name]
    layers = ["--layers", ",".join(str(L) for L in range(m.n_layers - 1))]
    print(f"{key}: fitting source layers 0-{m.n_layers - 2} on {num_prompts} prompts (router on {router_prompts})", flush=True)

    # fit-weights reads the correctness column: the expert weights are fitted only on items the model answers itself
    step("grade_correctness", d / "correctness.csv", ["grade-correctness", *hf, "--out", str(d / "correctness.csv")], d, timings)
    correctness = str(merged_correctness(fits_dir))

    router = d / "router.pt"
    step("fit_router", router, ["fit-router", *hf, *layers, "--num-prompts", str(router_prompts), "--out", str(router)], d, timings)

    artifacts = d / "artifacts"
    name = f"{PREFIX[key]}_experts_n{num_prompts}"
    done = d / f"{name}.shard_done"  # the shard writes to artifacts/<date>/<name>/shard0of1/, so a marker records that it finished
    shard_cmd = ["fit-shard", *hf, *layers, "--router-path", str(router), "--num-prompts", str(num_prompts), "--lrp-mode", "rlens"]
    step("fit_shard", done, [*shard_cmd, "--artifacts-base-dir", str(artifacts), "--checkpoint-name", name], d, timings)
    shard_dirs = sorted(artifacts.glob(f"*/{name}/shard0of1"))
    if len(shard_dirs) != 1:
        raise SystemExit(f"expected one shard directory for {name} under {artifacts}, found {shard_dirs}")
    done.write_text(str(shard_dirs[0]) + "\n")

    experts = d / "experts.pt"
    rlens, jpp = lens_paths(fits_dir, key, num_prompts)
    merge_cmd = ["merge-experts", "--checkpoint-dirs", str(shard_dirs[0]), "--checkpoint-name", name, "--out", str(experts)]
    step("merge_experts", experts, [*merge_cmd, "--pooled-lens-out", str(rlens)], d, timings)
    weights_cmd = ["fit-weights", *hf, *layers, "--experts", str(experts), "--items", "fit:0", "--correctness-csv", correctness]
    step("fit_weights", jpp, [*weights_cmd, "--checkpoint-name", jpp.stem, "--out", str(jpp)], d, timings)
    print(f"{key}: {rlens.name}, {jpp.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="*", default=list(FITTABLE), help=f"models to fit (default: {' '.join(FITTABLE)})")
    p.add_argument("--fits", required=True, help="output directory")
    p.add_argument("--num-prompts", type=int, default=64, help="prompts for the Jacobian sums: 64 is the post's and Neuronpedia's (default 64)")
    p.add_argument("--router-prompts", type=int, default=1000, help="prompts for the k-means router, the harness's default (default 1000)")
    args = p.parse_args()
    for key in args.keys:
        if key not in FITTABLE:
            raise SystemExit(f"{key} cannot be fitted here; the fittable models are {FITTABLE}")
    fits_dir = Path(args.fits).resolve()
    fits_dir.mkdir(parents=True, exist_ok=True)
    record_environment(fits_dir)
    for key in args.keys:
        fit_model(key, fits_dir, args.num_prompts, args.router_prompts)


if __name__ == "__main__":
    main()
