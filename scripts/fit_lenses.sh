#!/usr/bin/env bash
# Fit the J-lenses of the audio models no published lens covers (config.FIT_MODELS), one model at a time on one GPU
# (scripts/fit_lens.py): first a 10-prompt timing run, whose meta.json projects the full fit's hours, then the full fit with N
# prompts, unless TIMING_ONLY=1. A step whose lens.pt exists is skipped; an interrupted fit resumes from its checkpoint.
#
#   TIMING_ONLY=1 scripts/fit_lenses.sh                      the three timing runs: decide N and the GPUs from them
#   scripts/fit_lenses.sh                                    all three, 250 prompts each
#   MODELS="Qwen/Qwen2.5-Omni-3B" N=250 scripts/fit_lenses.sh
#   MODELS="google/gemma-4-26B-A4B-it" scripts/fit_lenses.sh  the mid-size Gemma-4 for video (51.6 GB: an H200), timing then 250
#   MODELS="google/gemma-4-12B-it" TIMING_ONLY=1 scripts/fit_lenses.sh   the dense 12B's 10-prompt test first (written before it
#       runs: the full fit goes ahead only if the J-lens agrees with the model on at least 0.5 of the report's positions at the
#       deepest fitted layer; the three omni models' 10-prompt lenses gave 0.63, the 12B's published lens 0.14 on check_lens)
#   CHECK_RECIPE=1 scripts/fit_lenses.sh                     the cheap checks first (config.LENS_CHECKS, about 1 hour on one H200):
#       1. this recipe refits Qwen3.5-2B with 283 prompts (runs/lens_repro_fit) and is compared with Neuronpedia's lens of it
#          (runs/lens_repro); 3's calibration: Neuronpedia's Qwen3.5-2B, 4B and 27B J-lenses' concept readout (runs/lens_readout_np)
#   EXTEND_TO=1000 scripts/fit_lenses.sh                     each lens in runs/fit_lens (250 prompts) extended to 1000, only after
#       check 1 passed: prompts 250-499 (runs/fit_lens_ext500) and 500-999 (runs/fit_lens_ext1000b) are fitted apart, so the
#       lens's halves exist (0-499: runs/fit_lens_h0), and all three merge into runs/fit_lens_n1000, the lens one 1000-prompt fit
#       gives. Then check 2 (runs/lens_halves), check 3 (runs/lens_readout) and the verdict (runs/lens_validation).
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
EXTEND_TO=${EXTEND_TO:-}
CHECK_RECIPE=${CHECK_RECIPE:-0}

uv sync --frozen
if [ "$CHECK_RECIPE" = 1 ]; then
    ref=$(uv run --frozen python -c "from jspace.config import LENS_CHECKS as c; print(c['reproduce'], c['reproduce_n'])")
    rm_=${ref% *}; rn=${ref#* }
    uv run --frozen scripts/download.py --skip-data --models "$rm_"
    [ -f "runs/lens_repro_fit/${rm_#*/}/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$rm_" --n "$rn" --run lens_repro_fit --dim-batch "$DIM_BATCH"
    uv run --frozen scripts/compare_lenses.py --model "$rm_" --a pinned --b runs/lens_repro_fit --run lens_repro || true
    for c in $(uv run --frozen python -c "from jspace.config import READOUT_CALIBRATION as r; print(' '.join(r))"); do
        uv run --frozen scripts/download.py --skip-data --models "$c"
        [ -f "runs/lens_readout_np/${c#*/}/summary.json" ] || uv run --frozen scripts/lens_readout.py --model "$c" --lens pinned --run lens_readout_np
    done
    # check 3's bar, and whether it is informative (every calibration lens beats its own logit lens), known before any fit
    uv run --frozen python -c "
import json
from jspace import config, lenscheck
m = {c: lenscheck.margin(json.load(open(f'runs/lens_readout_np/{c.split(\"/\")[-1]}/summary.json'))) for c in config.READOUT_CALIBRATION}
bar, ok = lenscheck.readout_bar(m)
print('check 3: margins over the logit lens', {c: round(v, 3) for c, v in m.items()}, 'bar', round(bar, 3), '->', 'informative' if ok else 'VOID: no fitted lens can pass check 3')
" || true
    exit 0
fi
for m in $MODELS; do
    tag=${m#*/}
    if [ -n "$EXTEND_TO" ]; then  # the existing lens's prompts are 0-249; fit 250-499 and 500-999 apart, merge, then check it
        [ "$EXTEND_TO" = 1000 ] || { echo "EXTEND_TO=1000 only: its halves are config.LENS_CHECKS's" >&2; exit 1; }
        uv run --frozen python -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['passes'] else 1)" "runs/lens_repro/Qwen3.5-2B/compare.json" ||
            { echo "check 1 has not passed (runs/lens_repro: run CHECK_RECIPE=1 first): nothing more is fitted" >&2; exit 1; }
        [ -f "runs/fit_lens/$tag/meta.json" ] || { echo "$m: no lens in runs/fit_lens to extend" >&2; exit 1; }
        n0=$(uv run --frozen python -c "import json, sys; m = json.load(open(sys.argv[1])); assert not m.get('shard') and not m.get('start'), m; print(m['n_requested'])" "runs/fit_lens/$tag/meta.json")
        [ "$n0" = 250 ] || { echo "$m: its lens has $n0 prompts, not 250" >&2; exit 1; }
        [ -f "runs/fit_lens_ext500/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n 500 --start 250 --run fit_lens_ext500 --dim-batch "$DIM_BATCH"
        [ -f "runs/fit_lens_ext1000b/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n 1000 --start 500 --run fit_lens_ext1000b --dim-batch "$DIM_BATCH"
        [ -f "runs/fit_lens_h0/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --merge runs/fit_lens runs/fit_lens_ext500 --run fit_lens_h0
        [ -f "runs/fit_lens_n1000/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --merge runs/fit_lens runs/fit_lens_ext500 runs/fit_lens_ext1000b --run fit_lens_n1000 ${PUSH_TO:+--push-to "$PUSH_TO" --push-dir "$tag/n1000"}
        uv run --frozen scripts/compare_lenses.py --model "$m" --a runs/fit_lens_h0 --b runs/fit_lens_ext1000b --run lens_halves || true
        [ -f "runs/lens_readout/$tag/summary.json" ] || uv run --frozen scripts/lens_readout.py --model "$m" --lens runs/fit_lens_n1000 --run lens_readout || true
        uv run --frozen scripts/validate_lens.py --model "$m" --lens runs/fit_lens_n1000 || true
        continue
    fi
    [ -f "runs/fit_lens_timing/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n 10 --run fit_lens_timing --dim-batch "$DIM_BATCH"
    uv run --frozen python -c "import json, sys; m = json.load(open(sys.argv[1])); print(sys.argv[2], round(m['seconds_per_prompt'], 1), 's per prompt; projected hours on one GPU', m['projected_hours_one_gpu'])" "runs/fit_lens_timing/$tag/meta.json" "$m"
    [ "$TIMING_ONLY" = 1 ] && continue
    [ -f "runs/fit_lens/$tag/lens.pt" ] || uv run --frozen scripts/fit_lens.py --model "$m" --n "$N" --run fit_lens --dim-batch "$DIM_BATCH" ${PUSH_TO:+--push-to "$PUSH_TO"}
done
