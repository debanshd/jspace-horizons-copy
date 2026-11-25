#!/usr/bin/env bash
# Calibration of scripts/check_lens.py's rule on Qwen: does it pass the Neuronpedia lenses the original Qwen results read? Every
# Qwen lens checked so far has failed it (the 9B J-lens, the omni 250-prompt lenses), all by losing to the plain logit lens, and
# the rule was never run on a Qwen lens known to work. Into runs/check_lens_calibration, each against Gemma-4-E4B's lens run there
# too: Qwen3.5-2B, 4B and 27B on their Neuronpedia J-lenses, Qwen3.5-9B on its Neuronpedia J++ lens (its pinned lens).
#
# What it decides (written before the run):
# - all four pass: the rule stands for Qwen. The 9B J-lens's failure is the lens's, and the 1000-prompt omni extension
#   (PR #12) is worth its cost; it still needs its own go-ahead.
# - any fails: the rule is not calibrated for Qwen. No Qwen lens is judged by it, and a new rule is written before any Qwen
#   lens check is rerun; the 1000-prompt extension waits for that rule.
# Either way the 9B J-lens result stands as recorded in runs/check_lens, and results/ is untouched (its lenses were never gated).
#
#   scripts/lens_calibration.sh            one H200 (the 27B), about 20 minutes with the downloads
set -uo pipefail
cd "$(dirname "$0")/.."
REF=${REF:-google/gemma-4-E4B-it}
MODELS=${MODELS:-"Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B"}
RUN=check_lens_calibration

uv sync --frozen || exit 1
uv run --frozen scripts/download.py --skip-data --models "$REF" || exit 1
[ -f "runs/$RUN/${REF#*/}/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$REF" --run "$RUN" || exit 1
for m in $MODELS; do
    tag=${m#*/}
    echo "===== $m"
    uv run --frozen scripts/download.py --skip-data --models "$m" || { echo "$m: download failed" >&2; continue; }
    if [ -f "runs/$RUN/$tag/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$m" --run "$RUN" --reference "$REF"; then
        uv run --frozen python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: s.get(k) for k in ('deepest', 'beats_logit_lens', 'near_reference', 'passes')}, 'J-lens', s['jlens'], 'logit lens', s['logit'])" "runs/$RUN/$tag/summary.json"
    else
        echo "$m: check_lens did not run to the end: see runs/$RUN/$tag/run.log" >&2
    fi
done
