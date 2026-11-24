#!/usr/bin/env bash
# The lens checks to run now, and nothing else: scripts/check_lens.py on every lens the experiments would read that is not
# Neuronpedia's own, each against Gemma-4-E4B's Neuronpedia lens. No experiment runs here.
#   1. Qwen3.5-9B's J-lens (bcywinski/jacobian-lens-qwen3.5-9b, --lens jlens, PR #15) into runs/check_lens: the deciding check
#      for the mid-size video runs; scripts/video_sweep2.sh reads the 9B through it only if this passes.
#   2. The Qwen omni models on their 250-prompt lenses into runs/check_lens_n250 (scripts/omni_first_pass.sh): a loading test
#      whose pass/fail decides nothing; their deciding lenses are the 1000-prompt ones (config.DECIDING_LENS), checked later.
# Not yet: Gemma-4-12B's and Gemma-4-26B-A4B's lenses (PR #12), which do not exist until they are fitted.
#
#   scripts/lens_checks.sh            one H200 (Qwen3-Omni-30B-A3B needs it), about 25 minutes
set -uo pipefail
cd "$(dirname "$0")/.."
REF=${REF:-google/gemma-4-E4B-it}

uv sync --frozen || exit 1
uv run --frozen scripts/download.py --skip-data --models "$REF" || exit 1
[ -f "runs/check_lens/${REF#*/}/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$REF" --run check_lens || exit 1

echo "===== Qwen/Qwen3.5-9B, lens jlens"
uv run --frozen scripts/download.py --skip-data --models Qwen/Qwen3.5-9B --lens jlens || echo "Qwen3.5-9B: download failed" >&2
if [ -f runs/check_lens/Qwen3.5-9B/summary.json ] || uv run --frozen scripts/check_lens.py --model Qwen/Qwen3.5-9B --lens jlens --run check_lens --reference "$REF"; then
    uv run --frozen scripts/audio_gate.py lens runs/check_lens/Qwen3.5-9B/summary.json || true  # prints the numbers; decides in video_sweep2.sh
else
    echo "Qwen3.5-9B: check_lens did not run to the end: see runs/check_lens/Qwen3.5-9B/run.log" >&2
fi

scripts/omni_first_pass.sh
