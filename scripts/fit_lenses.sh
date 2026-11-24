#!/usr/bin/env bash
# Fit the J-lenses of the audio models no published lens covers (config.FIT_MODELS), one model at a time on one GPU
# (scripts/fit_lens.py): first a 10-prompt timing run, whose meta.json projects the full fit's hours, then the full fit with N
# prompts, unless TIMING_ONLY=1. A step whose lens.pt exists is skipped; an interrupted fit resumes from its checkpoint.
#
#   TIMING_ONLY=1 scripts/fit_lenses.sh                      the three timing runs: decide N and the GPUs from them
#   scripts/fit_lenses.sh                                    all three, 250 prompts each
#   MODELS="Qwen/Qwen2.5-Omni-3B" N=250 scripts/fit_lenses.sh
#
# GPUs: Qwen2.5-Omni-3B and 7B fit on an L40S (48 GB). Qwen3-Omni-30B-A3B's text part alone is about 60 GB in bf16: an H200
# (141 GB). PUSH_TO=<a Hugging Face model repo> uploads each lens with its meta.json (a write token in .env).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct"}
N=${N:-250}
TIMING_ONLY=${TIMING_ONLY:-0}
DIM_BATCH=${DIM_BATCH:-8}
PUSH_TO=${PUSH_TO:-}

uv sync --frozen
for m in $MODELS; do
    tag=${m#*/}
    [ -f "runs/fit_lens_timing/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n 10 --run fit_lens_timing --dim-batch "$DIM_BATCH"
    uv run --frozen python -c "import json, sys; m = json.load(open(sys.argv[1])); print(sys.argv[2], round(m['seconds_per_prompt'], 1), 's per prompt; projected hours on one GPU', m['projected_hours_one_gpu'])" "runs/fit_lens_timing/$tag/meta.json" "$m"
    [ "$TIMING_ONLY" = 1 ] && continue
    [ -f "runs/fit_lens/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n "$N" --run fit_lens --dim-batch "$DIM_BATCH" ${PUSH_TO:+--push-to "$PUSH_TO"}
done
