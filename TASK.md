# TASK.md

The current task for an agent or a new contributor. The rules are in [AGENTS.md](AGENTS.md).

## Objective

Reproduce the results on your hardware and report how close the reproduction is.

## Steps

1. `uv sync`, then `uv run pytest -q`: every test passes.
2. `uv run scripts/download.py` (add `--gemma` only with a GPU of more than 90 GB): every data and lens file's sha256 matches.
3. One small run: `uv run experiments/depth_sweep.py --n 4 --run try_depth_sweep`.
4. `scripts/reproduce.sh`, or `GEMMA=1 scripts/reproduce.sh` with a GPU of more than 90 GB.

## Acceptance

- `scripts/check.py` reports every produced output as `same`. On another GPU type, each difference is listed (file, column, rows), with whether any verdict in a `summary.json` changes.
- `runs/examples/` holds the figures and `outputs.csv`.
- Each run's `config.json` records the GPU and the package versions.

## Permitted changes

None to the scientific invariants in AGENTS.md. Fixes to the environment (installation, paths) are allowed and go in the report.

## Deliverable

A short note: the hardware, the time per run, the output of `scripts/check.py`, and every difference with its cause.

## Open questions for later tasks

These come from the report's loose ends; none has been run.

1. Do the horizons predict failures on natural photos? On GQA the lens gives the layer at which each question's object is first named on its tokens. If the objects named after the fact horizon are the ones whose attribute and relation questions fail while naming questions succeed, one lens reading predicts which questions a model gets wrong, without any patching.
2. Which attention heads read early and which late? Knocking out attention from the question tokens to the image tokens, head by head, would show whether the rule readers sit before the fact horizon and the naming readers before the name horizon, and why Gemma's fact horizon is later. In Qwen3.5 this applies cleanly only to the one layer in four that uses standard attention.
3. Removing the answer word's single lens direction from a GQA photo costs about 2600 / width nats across Qwen3.5 0.8B, 2B and 4B. Does that hold when every model ablates the direction at the same share of its layers, and at 27B?
