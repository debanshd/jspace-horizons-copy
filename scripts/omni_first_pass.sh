#!/usr/bin/env bash
# What PR #14 runs for now: check_lens on the Qwen omni models, nothing else. No audio experiment runs (not audio_clean, not the
# chain); those come later, on the 1000-prompt lenses, by a separate decision.
# Per model, on its 250-prompt lens: scripts/check_lens.py into runs/check_lens_n250, beside Gemma-4-E4B's Neuronpedia lens (run
# there too). Session loads the thinker, its picture check must read the picture (the full thinker, model.entry_points), and the
# lens is read at every read layer against the model's own prediction. Its pass/fail decides nothing: the deciding lenses are
# the 1000-prompt ones (config.DECIDING_LENS, written before this ran), which get their own check_lens in runs/check_lens.
# A model that fails to load stops with its error; the next model still runs.
#
#   scripts/omni_first_pass.sh            all three on one H200 (Qwen3-Omni-30B-A3B needs it; the two 2.5s fit on an L40S)
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct"}
REF=${REF:-google/gemma-4-E4B-it}
RUN=check_lens_n250

uv sync --frozen || exit 1
uv run --frozen scripts/download.py --skip-data --models "$REF" || exit 1
[ -f "runs/$RUN/${REF#*/}/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$REF" --run "$RUN" || exit 1
for m in $MODELS; do
    tag=${m#*/}
    echo "===== $m"
    uv run --frozen scripts/download.py --skip-data --models "$m" || { echo "$m: download failed" >&2; continue; }
    if [ -f "runs/$RUN/$tag/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$m" --run "$RUN" --reference "$REF"; then
        uv run --frozen python -c "import json, sys; s = json.load(open(sys.argv[1])); print({k: s.get(k) for k in ('deepest', 'beats_logit_lens', 'near_reference', 'passes')}, 'J-lens by layer', s['jlens'])" "runs/$RUN/$tag/summary.json"
    else
        echo "$m: check_lens did not run to the end (loading or the picture check): see runs/$RUN/$tag/run.log" >&2
        continue
    fi
    # the same lens by the Qwen rule (config.QWEN_CALIBRATION), into check_lens_qwen_n250: a preview of what the 1000-prompt
    # lens faces; this decides nothing either
    [ -f "runs/check_lens_qwen_n250/$tag/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$m" --run check_lens_qwen_n250 --calibration check_lens_calibration ||
        { echo "$m: the Qwen-rule check_lens did not run to the end: see runs/check_lens_qwen_n250/$tag/run.log" >&2; continue; }
    uv run --frozen scripts/audio_gate.py lens "runs/check_lens_qwen_n250/$tag/summary.json" || true
done
