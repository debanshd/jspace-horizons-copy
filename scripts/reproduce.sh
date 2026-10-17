#!/usr/bin/env bash
# Every run behind the figures, in the order they depend on each other, then the figures and the comparison
# with results/. A run whose summary.json exists is skipped, so the script picks up where it stopped.
#
#   scripts/reproduce.sh            the Qwen3.5-4B runs (about an hour of GPU time); the Gemma panel is drawn from results/
#   GEMMA=1 scripts/reproduce.sh    also the two Gemma-4-31B runs (a 69 GB download, 45 minutes, a GPU of more than 90 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
Q=Qwen/Qwen3.5-4B
G=google/gemma-4-31B-it

run() {  # run <run name> <experiment> <model> [arguments]
    local name=$1 exp=$2 model=$3
    shift 3
    if [ -f "runs/$name/${model#*/}/summary.json" ]; then
        echo "done already: runs/$name/${model#*/}"
        return
    fi
    uv run "experiments/$exp.py" --model "$model" --run "$name" "$@"
}

if [ "${GEMMA:-0}" = 1 ]; then uv run scripts/download.py --gemma; else uv run scripts/download.py; fi   # checked, resumable

run printed_word         printed_word      $Q                                # written words
run scam_notes           scam_notes        $Q                                # written words
run name_steering        name_steering     $Q                                # steering
run subspace_patching    subspace_patching $Q --ranks 16,64,256,1024         # subspace patching; its stimulus sheet supplies the figures' canvases
run depth_sweep          depth_sweep       $Q                                # back_patching reads its horizons
run depth_sweep_spectrum depth_sweep       $Q --spectrum                     # the five-question depth figure
run back_patching        back_patching     $Q                                # back-patching
run gqa_ablation         gqa_ablation      $Q                                # GQA photos
if [ "${GEMMA:-0}" = 1 ]; then
    run subspace_patching subspace_patching $G                               # checks Gemma saw the same canvases
    run depth_sweep       depth_sweep       $G                               # the Gemma panel of the depth figure
fi
uv run scripts/make_figures.py --fallback results
uv run scripts/check.py
