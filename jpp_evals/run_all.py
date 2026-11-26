"""Run the J++ post's readout and probe-swap evals on the manifest's models, one model after another, and write one summary.

Per model, in <out>/<model key>/:
- correctness.csv: which items the model answers itself (run_cli.py grade-correctness). The readout evals keep multihop,
  multilingual and poetry items only where the model is right (typo and association are always kept). For Qwen3.6-27B the
  post's own column is used, and this grading is kept beside it as a check of the pipeline.
- readout_filtered*.csv / readout_full*.csv: the readout evals with Readout Filtering (tokens with no letter or digit out of the
  ranking, as the post scores J++) and over the full vocabulary (as the post scores the J-lens and R-lens); every lens of the model
  and the logit lens.
- probe_swap*.parquet / *_success.csv / *_best_scale.csv: the probe-swap causal eval, every lens and the logit lens.
- timings.json: the wall time of each step, and *.log: each step's output.

Then <out>/summary.csv and <out>/summary.md. A step whose output exists is skipped, so a stopped run resumes where it stopped.

Run from the root of a jpp_lens checkout, in its environment: uv run python <repo>/jpp_evals/run_all.py --out <dir> [keys ...]
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import JPP_LENS_COMMIT, MODELS, Lens, Model, model_by_key  # noqa: E402

HERE = Path(__file__).resolve().parent
SHIPPED_CORRECTNESS = Path("data/jlens/evaluations/model_correctness.csv")
MACRO_EVALS = ["multihop", "multilingual", "typo", "association", "poetry"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch_lens(lens: Lens, lens_dir: Path) -> Path:
    """the lens file at its pinned commit, checked against its sha256, linked as <lens.name>.pt (the harness names a lens by its stem)"""
    from huggingface_hub import hf_hub_download

    out = lens_dir / f"{lens.name}.pt"
    if out.exists():
        return out
    repo, revision = lens.repo
    cached = Path(hf_hub_download(repo, lens.file, revision=revision))
    got = sha256(cached)
    if got != lens.sha256:
        raise SystemExit(f"{repo}@{revision}:{lens.file} has sha256 {got}, the manifest pins {lens.sha256}")
    lens_dir.mkdir(parents=True, exist_ok=True)
    out.symlink_to(cached)
    return out


def step(name: str, out: Path, cmd: list[str], model_dir: Path, timings: dict) -> bool:
    """run one stage unless its output exists from the same command; its output goes to <name>.log, its wall time to timings.json.

    The command (lenses, items, layers, correctness file) is recorded in <name>.cmd.json when the stage succeeds, and an existing
    output counts as done only when that record matches: a second fit scored into the same folder changes the lens list, so its
    readout and probe-swap must run again rather than be skipped. Returns whether the stage ran."""
    record = model_dir / f"{name}.cmd.json"
    if out.exists() and record.exists() and json.loads(record.read_text()) == cmd:
        print(f"  {name}: done before, skipped")
        return False
    record.unlink(missing_ok=True)
    print(f"  {name}: running", flush=True)
    t0 = time.time()
    with open(model_dir / f"{name}.log", "w") as log:
        rc = subprocess.run([sys.executable, str(HERE / "run_cli.py"), *cmd], stdout=log, stderr=subprocess.STDOUT).returncode
    timings[name] = round(time.time() - t0, 1)
    (model_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    if rc != 0:
        tail = (model_dir / f"{name}.log").read_text().splitlines()[-30:]
        raise SystemExit(f"{name} failed (exit {rc}); the end of {model_dir / f'{name}.log'}:\n" + "\n".join(tail))
    record.write_text(json.dumps(cmd))
    print(f"  {name}: {timings[name]:.0f} s", flush=True)
    return True


def merged_correctness(out_dir: Path, m: Model) -> Path:
    """the correctness CSV for one model: the shipped CSV, with this model's graded column added unless the shipped CSV has it (the
    post's own column is kept). One file per model, because the harness names the column by the repo, and two checkpoints of one
    repo (OLMo at 100% and 50%) answer differently."""
    df = pd.read_csv(SHIPPED_CORRECTNESS)
    if m.hf_name not in df.columns:
        df = df.merge(pd.read_csv(out_dir / m.key / "correctness.csv"), on=["eval", "item"], how="left")
    path = out_dir / m.key / "model_correctness.csv"
    df.to_csv(path, index=False)
    return path


def fitted_lenses(fits_dir: Path | None, key: str) -> list[Path]:
    """the R-lens and J++ lens files fit_lenses.py wrote for this model, if any"""
    if fits_dir is None:
        return []
    d = fits_dir / key
    return sorted(d.glob("*_rlens_n*.pt")) + sorted(d.glob("*_jpp_n*.pt"))


def run_model(m: Model, out_dir: Path, prune: bool, fits_dir: Path | None = None) -> None:
    print(f"{m.key} ({m.hf_name}@{m.revision[:7]}), layers {m.layers}", flush=True)
    model_dir = out_dir / m.key
    model_dir.mkdir(parents=True, exist_ok=True)
    timings_path = model_dir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    t0 = time.time()
    lens_paths = [str(fetch_lens(lens, out_dir / "lenses")) for lens in m.lenses]
    timings.setdefault("fetch_lenses", round(time.time() - t0, 1))
    fitted = fitted_lenses(fits_dir, m.key)
    lens_paths += [str(f) for f in fitted]
    items_spec = m.items
    reuse_fit_grading = any("_jpp_" in f.name for f in fitted)
    if reuse_fit_grading:
        # The J++ expert weights were fitted on the "fit:0" split, so every lens of this model is scored on the rest. The split is
        # drawn from the items the correctness column keeps, so the scoring must read the very file fit-weights read: no grading
        # here (a regrade, even of the same model, could tip a borderline item and move the split).
        items_spec = "held-out:0"
    if fitted:
        print(f"  with the fitted lenses {[f.stem for f in fitted]}, scored on {items_spec}", flush=True)

    common = ["--model-key", m.key, "--hf-model-name", m.hf_name]
    layers = ["--layers", ",".join(map(str, m.layers))]
    if reuse_fit_grading:
        fit_correctness = fits_dir / m.key / "model_correctness.csv"
        if not fit_correctness.exists():
            raise SystemExit(f"{fit_correctness} is missing: the J++ lens's fit wrote it, and its split depends on it")
        shutil.copyfile(fits_dir / m.key / "correctness.csv", model_dir / "correctness.csv")  # kept beside the scores, for reading
        correctness = str(fit_correctness)
    else:
        step("grade_correctness", model_dir / "correctness.csv", ["grade-correctness", *common, "--out", str(model_dir / "correctness.csv")], model_dir, timings)
        correctness = str(merged_correctness(out_dir, m))
    lenses = ["--lens", *lens_paths, "logit"]
    items = ["--items", items_spec, "--correctness-csv", correctness]
    for mode, extra in (("filtered", []), ("full", ["--no-readout-filter"])):
        out = model_dir / f"readout_{mode}.csv"
        step(f"readout_{mode}", out, ["evaluate", *common, *layers, *lenses, *items, *extra, "--out", str(out)], model_dir, timings)
    out = model_dir / "probe_swap.parquet"
    step("probe_swap", out, ["probe-swap", *common, *layers, *lenses, "--out", str(out)], model_dir, timings)

    if prune:  # the 30B and 27B checkpoints alone fill most of a 200 GB volume
        from huggingface_hub import scan_cache_dir

        # This model's revision only: OLMo's checkpoints share one repo, and deleting all its revisions would make the next
        # checkpoint download again. Files another cached revision shares are kept by the cache.
        cache = scan_cache_dir()
        hashes = [r.commit_hash for repo in cache.repos if repo.repo_id == m.hf_name for r in repo.revisions if r.commit_hash == m.revision]
        if hashes:
            strategy = cache.delete_revisions(*hashes)
            print(f"  pruned {m.hf_name}@{m.revision[:7]} from the cache ({strategy.expected_freed_size_str})")
            strategy.execute()


def markdown_table(df: pd.DataFrame) -> list[str]:
    """a Markdown table with one decimal (DataFrame.to_markdown needs tabulate, which the harness's environment does not have)"""

    def cell(v) -> str:
        return f"{v:.1f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return lines


def split_check(m: Model, model_dir: Path) -> str | None:
    """how the items this run scored compare with the split a lens's expert weights were fitted on, where that split is published.
    The correctness grading decides which items enter the split, and a grading with another attention kernel (bubskee graded with
    eager, this run with sdpa) can tip a borderline item, so a difference is reported rather than fatal."""
    stamp = model_dir / "readout_filtered_stamp.json"
    if m.split_reference is None or not stamp.exists():
        return None
    from huggingface_hub import hf_hub_download

    repo, revision, file = m.split_reference
    ref = json.loads(Path(hf_hub_download(repo, file, revision=revision)).read_text())
    fit = {name for _, name in ref["fit"]}
    heldout = {name for _, name in ref["heldout"]}
    scored = set(json.loads(stamp.read_text())["scored_item_names"])
    return (
        f"{m.key}: {len(scored)} items scored; {len(scored & heldout)} of the {len(heldout)} items {repo} holds out, and "
        f"{len(scored & fit)} of the {len(fit)} its J++ weights were fitted on (0 expected); "
        f"{len(scored - fit - heldout)} scored items are in neither list."
    )


def summarize(out_dir: Path, keys: list[str]) -> None:
    """one row per (model, lens): macro recall@1 and @10 with and without Readout Filtering, recall@10 per eval, probe-swap at the
    best scale"""
    rows = []
    for key in keys:
        m = model_by_key(key)
        d = out_dir / key
        if not (d / "probe_swap_best_scale.csv").exists():
            continue
        recall = {mode: pd.read_csv(d / f"readout_{mode}_pass_at_k.csv") for mode in ("filtered", "full")}
        swap = pd.read_csv(d / "probe_swap_best_scale.csv").set_index("lens")
        for lens in recall["filtered"]["lens"].unique():  # the manifest's lenses, any fitted ones, and the logit lens
            row = {"model": key, "lens": lens}
            for mode, df in recall.items():
                df = df[(df["lens"] == lens) & (df["weighting"] == "item")]
                for k in (1, 10):
                    row[f"{mode}_macro_r@{k}"] = 100 * df[(df["eval"] == "macro") & (df["k"] == k)]["pass_at_k"].item()
                if mode == "filtered":
                    for ev in MACRO_EVALS:
                        sel = df[(df["eval"] == ev) & (df["k"] == 10)]["pass_at_k"]
                        row[f"{ev}_r@10"] = 100 * sel.item() if len(sel) else float("nan")
            s = swap.loc[lens]
            row |= {
                "swap_top1": 100 * s["top1_rate"],
                "swap_top1_hits": f"{s['top1_hits']}/{s['top1_trials']}",
                "swap_top5": 100 * s["top5_rate"],
                "swap_scale": s["scale"],
            }
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    lines = ["# J++ post evals: omni J-lenses and Neuronpedia baselines", ""]
    lines += [f"jpp_lens {JPP_LENS_COMMIT[:7]}. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at", ""]
    lines += ["the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary", ""]
    lines += ["(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.", ""]
    lines += markdown_table(df)
    for key in keys:
        m = model_by_key(key)
        graded = out_dir / key / "correctness.csv"
        if graded.exists() and m.hf_name in pd.read_csv(SHIPPED_CORRECTNESS).columns:
            ours = pd.read_csv(graded).merge(pd.read_csv(SHIPPED_CORRECTNESS)[["eval", "item", m.hf_name]], on=["eval", "item"], suffixes=("", "_post"))
            both = ours.dropna(subset=[m.hf_name, f"{m.hf_name}_post"])
            agree = int((both[m.hf_name].astype(bool) == both[f"{m.hf_name}_post"].astype(bool)).sum())
            lines += ["", f"{key}: this run's correctness grading agrees with the post's on {agree} of {len(both)} gradeable items."]
    for key in keys:
        check = split_check(model_by_key(key), out_dir / key)
        if check:
            lines += ["", check]
    lines += ["", "Wall time per step, seconds:", ""]
    for key in keys:
        t = out_dir / key / "timings.json"
        if t.exists():
            lines.append(f"- {key}: " + ", ".join(f"{k} {v:.0f}" for k, v in json.loads(t.read_text()).items()))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def record_environment(out_dir: Path) -> None:
    import torch
    import transformers

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit != JPP_LENS_COMMIT:
        raise SystemExit(f"run from a jpp_lens checkout at {JPP_LENS_COMMIT}; this directory is at {commit or 'no git commit'}")
    # nvidia-smi rather than torch.cuda: a CUDA context in this long-lived parent would hold GPU memory while every stage runs
    gpus = None
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        gpus = smi.stdout.strip().splitlines() if smi.returncode == 0 else None
    config = {
        "jpp_lens_commit": commit,
        "gpu": gpus,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "argv": sys.argv,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="*", help="models to run (default: every model the manifest marks default)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--prune", action="store_true", help="delete each model from the Hugging Face cache once its evals are done")
    p.add_argument("--fits", default=None, help="also score the R-lens and J++ lenses fit_lenses.py wrote here")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    keys = args.keys or [m.key for m in MODELS if m.default]
    for key in keys:
        model_by_key(key)  # an unknown key stops the run before anything is downloaded
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        record_environment(out_dir)
        for key in keys:
            run_model(model_by_key(key), out_dir, args.prune, Path(args.fits).resolve() if args.fits else None)
    summarize(out_dir, keys)


if __name__ == "__main__":
    main()
