#!/usr/bin/env bash
# The first GPU pass on the Qwen omni models (PR #14), before their lenses are extended to 1000 prompts (PR #12): a loading test.
# Per model, on its 250-prompt lens:
#   1. scripts/check_lens.py into runs/check_lens_n250 (beside Gemma-4-E4B's Neuronpedia lens, run there too): Session loads the
#      thinker, its picture check must read the picture (the full thinker, model.entry_points), and the lens is read at every
#      read layer. Its pass/fail decides nothing: the deciding lenses are the 1000-prompt ones, decided before this pass ran
#      (config.DECIDING_LENS), and they get their own check_lens in runs/check_lens.
#   2. experiments/audio_clean.py into runs/audio_clean: the audio path (the processor, the audio token, generation) on every
#      recording and question. Its answers do not depend on the lens, so the chain later reads this run as its own.
# A model that fails to load or to run stops with its error; the next model still runs.
#
#   scripts/omni_first_pass.sh            all three on one H200 (Qwen3-Omni-30B-A3B needs it; the two 2.5s fit on an L40S)
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct"}
REF=${REF:-google/gemma-4-E4B-it}
RUN=check_lens_n250

uv sync --frozen || exit 1
uv run --frozen scripts/prepare_spoken.py || exit 1
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
    [ -f "runs/audio_clean/$tag/audio_clean.csv" ] || uv run --frozen experiments/audio_clean.py --model "$m" --run audio_clean || { echo "$m: audio_clean failed: see runs/audio_clean/$tag/run.log" >&2; continue; }
    uv run --frozen python -c "import json, sys; s = json.load(open(sys.argv[1])); print('named', {i: r['named'] for i, r in s['recognition'].items()}, 'ready', s['ready'])" "runs/audio_clean/$tag/summary.json"
done
