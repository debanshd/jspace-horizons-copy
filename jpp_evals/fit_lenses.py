"""Fit J++ lenses with the J++ post's recipe, in the post's harness: an R-lens and a J++ lens for Qwen2.5-Omni-7B's thinker, and a J++
lens only for OLMo 3 7B at 100% and at 50% of its training (manifest.py: the released model and stage1-step725000).

The stages are the harness's recipe, the one the post's Qwen3.6-27B lens and Neuronpedia's Qwen3.5-4B R-lens and J++ lens were fitted
with; the settings differ where noted:
- a router of k-means clusters (E = 8 per layer) on the activations of the first 1000 WikiText-103 prompts, the harness's default
  (fit-router). Neuronpedia's Qwen3.5-4B pair used 32 router prompts and eager attention (bubskee/qwen3.5-4b-r-jpp-lenses@c1b9747,
  provenance/fit-recipe.json); these fits use 1000 and sdpa, so that baseline does not share their router;
- expert Jacobian sums through the LRP backward pass ("rlens" rules), 128 tokens per prompt, the first 16 positions skipped
  (fit-shard), merged into the expert Jacobians (merge-experts);
- the R-lens: the pooled Jacobian of that fit, all experts together (merge-experts --pooled-lens-out);
- the J++ lens: the experts combined with weights fitted on the post's "fit:0" split of the readout items (fit-weights). A J++
  lens must then be scored on the held-out split only: run_all.py --fits does that.

How many prompts enter the sums, two ways:
- fixed (the default): the first 64 prompts, the post's and Neuronpedia's count for these lenses;
- --converge: Neuronpedia's J-lens rule, adapted. Its fits (fit_lens.py --n_prompts 1000 --min_prompts 100 --stop_window 10
  --stop_at_delta 0.002) stop once the mean, over the last 10 prompts, of the per-prompt relative change of the lens (averaged over
  layers) falls below 0.002, and never before 100 prompts. The harness fits in shards and has no per-prompt hook, so here the
  prompts go in blocks of --block (25) up to --max-prompts (1000), one fit-shard run per block; after each block the pooled
  Jacobian (the R-lens) is rebuilt from the summed shards and its relative change per layer, ||J_new - J_old||_F / ||J_new||_F,
  averaged over layers, is the block's change. Over a block of B prompts the running mean takes B steps in nearly independent
  directions, so the block's change is about sqrt(B) times a single step's: the rule compares change / sqrt(B) with 0.002, averaged
  over the last ceil(10 / B) blocks, and stops at the first block where that holds from 100 prompts on. Every block's change, and
  the mean change of the experts the J++ lens combines, go to convergence.csv, so another rule can be applied afterwards.

Every source layer is fitted (0 to n - 2; the target is the last block), as in the omni J-lenses, so the lenses serve this
repository's experiments as well as the post's seven readout layers. A backward pass yields every source layer at once, so the extra
layers cost memory for their sums, not passes.

The fittable models are Qwen2.5-Omni-7B (the default) and the two OLMo 3 7B checkpoints, named by key: the harness's LRP rules cover
OLMo 3 natively (Olmo3RMSNorm, Olmo3MLP; the post fitted Olmo 3 32B). Qwen2.5-Omni-3B keeps its J-lenses only. Qwen3-Omni-30B-A3B is
left out: its MoE thinker has classes the LRP surgery has no verified rules for.

Per model, in <fits>/<model key>/: correctness.csv, router.pt, artifacts/ (the shard sums), <name>_experts.pt,
<prefix>_rlens_n<N>.pt, <prefix>_jpp_n<N>.pt with its expert weights (N the prompts the sums saw), convergence.csv for a converged
fit, a log and the wall time of each step. A step whose output exists is skipped, so a stopped fit resumes where it stopped.

Run from the root of a jpp_lens checkout, in its environment: uv run python <repo>/jpp_evals/fit_lenses.py --fits <dir> [keys ...]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import model_by_key  # noqa: E402
from run_all import merged_correctness, record_environment, step  # noqa: E402

# model key -> (file prefix, whether to keep the R-lens). The R-lens is the pooled Jacobian of the same sums, so it costs nothing,
# but for OLMo only the J++ lens is wanted.
FITTABLE = {
    "qwen2.5-omni-7b": ("omni7b", True),
    "olmo-3-7b": ("olmo7b", False),
    "olmo-3-7b-pretrain-end": ("olmo7bs1end", False),
    "olmo-3-7b-50pct": ("olmo7b50pct", False),
}
PREFIX = {key: prefix for key, (prefix, _) in FITTABLE.items()}
DEFAULT_KEYS = ["qwen2.5-omni-7b"]
NUM_CLUSTERS = 8  # the harness's recipe; the shard checkpoint file is named by it
MIN_KEPT_POSITIONS = 50  # merge-experts' default: an expert with fewer positions falls back to the pooled Jacobian

# Neuronpedia's J-lens convergence rule (fit_lens.py, as their Qwen2.5-7B-Instruct and Qwen3.5-4B J-lenses were fitted)
NP_STOP_AT_DELTA = 0.002
NP_MIN_PROMPTS = 100
NP_STOP_WINDOW = 10


def shard_dir_of(ran: bool, done: Path, log: Path, name: str, tag: str) -> Path:
    """the directory a fit-shard run wrote. A shard's directory is named by the UTC date it started (artifacts/<date>/<name>/<tag>)
    and the harness never resumes one, so a shard that failed and ran again on a later day leaves two: the one that finished is the
    path fit-shard printed at the end of the log of the run that just ended, and a skipped shard's is in its done-marker."""
    if not ran:
        return Path(done.read_text().strip())
    printed = [line.strip() for line in log.read_text().splitlines() if line.strip().endswith(f"{name}/{tag}")]
    if not printed or not Path(printed[-1]).is_dir():
        raise SystemExit(f"{log} does not end with the directory of shard {tag} of {name}")
    done.write_text(printed[-1] + "\n")
    return Path(printed[-1])


class RunningSums:
    """the summed shard statistics, the pooled Jacobian per layer built from them, and the change each new shard makes"""

    def __init__(self):
        import torch

        self.torch = torch
        # CPU, not the GPU: this process stays alive while the next fit-shard and merge-experts run, and on the 7B these tensors
        # (the sums, the experts and the pooled Jacobians, ~24 GB, ~47 GB at the peak of add) would stay reserved on the GPU
        self.device = "cpu"
        self.sums = None
        self.pooled = None
        self.experts = None

    def add(self, checkpoint: Path) -> tuple[float, float]:
        """add one shard's sums; returns the pooled Jacobian's and the experts' mean relative change (nan for the first shard)"""
        from workspace_lens.fitting.types import ExpertFitSums

        state = self.torch.load(checkpoint, map_location=self.device, weights_only=True)
        shard = ExpertFitSums.from_state_dict(state)
        del state
        if self.sums is None:
            self.sums = shard
        else:
            self.sums.add_(shard)
        pooled, experts = {}, {}
        for layer, s in self.sums.layer_sums_L_dict.items():
            w = s.weight_sum_E.float()
            pooled[layer] = s.weighted_jacobian_row_sum_EFN.sum(0) / float(w.sum())
            kept = (s.position_count_E >= MIN_KEPT_POSITIONS) & (w > 0)
            experts[layer] = {int(e): s.weighted_jacobian_row_sum_EFN[e] / float(w[e]) for e in kept.nonzero().flatten()}
        change = (math.nan, math.nan)
        if self.pooled is not None:
            pooled_change = [self._rel(pooled[L], self.pooled[L]) for L in pooled]
            expert_change = [self._rel(J, self.experts[L][e]) for L in experts for e, J in experts[L].items() if e in self.experts[L]]
            change = (sum(pooled_change) / len(pooled_change), sum(expert_change) / max(1, len(expert_change)))
        self.pooled, self.experts = pooled, experts
        return change

    def _rel(self, new, old) -> float:
        return float(self.torch.linalg.matrix_norm(new - old) / self.torch.linalg.matrix_norm(new))


def fit_shards_until_converged(m, d: Path, hf, layers, router: Path, artifacts: Path, name: str, args, timings) -> tuple[list[Path], int]:
    """fit-shard one block of prompts at a time until Neuronpedia's rule (adapted to blocks, see the docstring) says the lens has
    converged, or --max-prompts is reached; returns the shard directories and the prompts they hold"""
    num_shards = args.max_prompts // args.block
    if num_shards * args.block != args.max_prompts:
        raise SystemExit(f"--max-prompts ({args.max_prompts}) must be a multiple of --block ({args.block})")
    window = math.ceil(NP_STOP_WINDOW / args.block)
    running = RunningSums()
    log_path = d / f"{name}.convergence.csv"
    log_rows = []
    shard_dirs = []
    for i in range(num_shards):
        done = d / f"{name}.shard{i}of{num_shards}_done"
        cmd = ["fit-shard", *hf, *layers, "--router-path", str(router), "--lrp-mode", "rlens"]
        cmd += ["--num-prompts", str(args.max_prompts), "--num-shards", str(num_shards), "--shard-idx", str(i)]
        ran = step(f"fit_shard{i}_{name}", done, [*cmd, "--artifacts-base-dir", str(artifacts), "--checkpoint-name", name], d, timings)
        shard_dir = shard_dir_of(ran, done, d / f"fit_shard{i}_{name}.log", name, f"shard{i}of{num_shards}")
        shard_dirs.append(shard_dir)

        n_prompts = (i + 1) * args.block
        pooled_change, expert_change = running.add(shard_dir / f"experts_K{NUM_CLUSTERS}_checkpoint.pt")
        per_prompt = pooled_change / math.sqrt(args.block)
        log_rows.append({"n_prompts": n_prompts, "block_change": pooled_change, "per_prompt_change": per_prompt, "expert_block_change": expert_change})
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            w.writeheader()
            w.writerows(log_rows)
        recent = [r["per_prompt_change"] for r in log_rows[-window:]]
        smoothed = sum(recent) / len(recent) if not any(math.isnan(x) for x in recent) else math.nan
        print(f"  {n_prompts} prompts: change per prompt {per_prompt:.5f}, smoothed {smoothed:.5f} (stop below {args.stop_at_delta})", flush=True)
        if n_prompts >= NP_MIN_PROMPTS and len(recent) == window and smoothed < args.stop_at_delta:
            print(f"  converged at {n_prompts} prompts", flush=True)
            return shard_dirs, n_prompts
    print(f"  not converged by {args.max_prompts} prompts: the lens is built from all of them", flush=True)
    return shard_dirs, args.max_prompts


def fit_model(key: str, fits_dir: Path, args) -> None:
    m = model_by_key(key)
    d = fits_dir / key
    d.mkdir(parents=True, exist_ok=True)
    timings_path = d / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    hf = ["--model-key", m.key, "--hf-model-name", m.hf_name]
    layers = ["--layers", ",".join(str(L) for L in range(m.n_layers - 1))]
    how = f"until converged, at most {args.max_prompts}" if args.converge else str(args.num_prompts)
    print(f"{key}: fitting source layers 0-{m.n_layers - 2} on {how} prompts (router on {args.router_prompts})", flush=True)

    # fit-weights reads the correctness column: the expert weights are fitted only on items the model answers itself
    step("grade_correctness", d / "correctness.csv", ["grade-correctness", *hf, "--out", str(d / "correctness.csv")], d, timings)
    correctness = str(merged_correctness(fits_dir, m))

    router = d / "router.pt"
    step("fit_router", router, ["fit-router", *hf, *layers, "--num-prompts", str(args.router_prompts), "--out", str(router)], d, timings)

    artifacts = d / "artifacts"
    if args.converge:
        name = f"{PREFIX[key]}_experts_conv{args.max_prompts}"
        shard_dirs, n_prompts = fit_shards_until_converged(m, d, hf, layers, router, artifacts, name, args, timings)
    else:
        name = f"{PREFIX[key]}_experts_n{args.num_prompts}"
        done = d / f"{name}.shard_done"  # the shard writes to artifacts/<date>/<name>/shard0of1/, so a marker records that it finished
        cmd = ["fit-shard", *hf, *layers, "--router-path", str(router), "--num-prompts", str(args.num_prompts), "--lrp-mode", "rlens"]
        ran = step(f"fit_shard_{name}", done, [*cmd, "--artifacts-base-dir", str(artifacts), "--checkpoint-name", name], d, timings)
        shard_dirs = [shard_dir_of(ran, done, d / f"fit_shard_{name}.log", name, "shard0of1")]
        n_prompts = args.num_prompts

    experts = d / f"{name}_experts.pt"
    rlens = d / f"{PREFIX[key]}_rlens_n{n_prompts}.pt"
    jpp = d / f"{PREFIX[key]}_jpp_n{n_prompts}.pt"
    merge_cmd = ["merge-experts", "--checkpoint-dirs", *map(str, shard_dirs), "--checkpoint-name", name, "--out", str(experts)]
    keep_rlens = FITTABLE[key][1]
    step(f"merge_{name}", experts, [*merge_cmd, *(["--pooled-lens-out", str(rlens)] if keep_rlens else [])], d, timings)
    weights_cmd = ["fit-weights", *hf, *layers, "--experts", str(experts), "--items", "fit:0", "--correctness-csv", correctness]
    step(f"fit_weights_{name}", jpp, [*weights_cmd, "--checkpoint-name", jpp.stem, "--out", str(jpp)], d, timings)
    print(f"{key}: {', '.join(f.name for f in ([rlens] if keep_rlens else []) + [jpp])}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="*", default=DEFAULT_KEYS, help=f"models to fit, of {' '.join(FITTABLE)} (default: {' '.join(DEFAULT_KEYS)})")
    p.add_argument("--fits", required=True, help="output directory")
    p.add_argument("--num-prompts", type=int, default=64, help="fixed fit: prompts for the Jacobian sums, the post's and Neuronpedia's 64 (default 64)")
    p.add_argument("--converge", action="store_true", help="fit blocks of prompts until Neuronpedia's convergence rule stops it")
    p.add_argument("--max-prompts", type=int, default=1000, help="--converge: the most prompts, Neuronpedia's 1000 (default 1000)")
    p.add_argument("--block", type=int, default=25, help="--converge: prompts per fit-shard run (default 25)")
    p.add_argument("--stop-at-delta", type=float, default=NP_STOP_AT_DELTA, help=f"--converge: Neuronpedia's threshold (default {NP_STOP_AT_DELTA})")
    p.add_argument("--router-prompts", type=int, default=1000, help="prompts for the k-means router, the harness's default (default 1000)")
    args = p.parse_args()
    for key in args.keys:
        if key not in FITTABLE:
            raise SystemExit(f"{key} cannot be fitted here; the fittable models are {list(FITTABLE)}")
    fits_dir = Path(args.fits).resolve()
    fits_dir.mkdir(parents=True, exist_ok=True)
    record_environment(fits_dir)
    for key in args.keys:
        fit_model(key, fits_dir, args)


if __name__ == "__main__":
    main()
