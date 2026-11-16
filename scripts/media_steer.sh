#!/usr/bin/env bash
# plan.md, steps 2 to 8, on one GPU machine (a RunPod pod). A step whose output exists is skipped, so the chain picks up where
# it stopped.
#
#   scripts/media_steer.sh                                      the two medium models (Qwen3.5-4B, Gemma-4-E4B): 48 GB is enough
#   STAGE=smoke scripts/media_steer.sh                          the 5-clip smoke check only (plan.md steps 5 and 6)
#   MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-27B google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it" \
#       scripts/media_steer.sh                                  all six (the 27B and the 31B need the 96 GB card)
#
# The gate runs for every model first, because the 64 clips are those that every model in the run names (plan.md).
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
RUN=${RUN:-media_steer}
STAGE=${STAGE:-main}
LIST=$(echo "$MODELS" | tr ' ' ,)

if ! uv run scripts/pin_media.py --check; then
    echo "pin the datasets and Gemma-4-E2B first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
    exit 1
fi
uv run scripts/download.py --skip-data --models "$LIST"           # the models and lenses, checked
uv run --with av==15.1.0 scripts/prepare_media.py                 # the clips, decoded once, checked on a re-run

if [ "$STAGE" = smoke ]; then
    for m in $MODELS; do
        [ -f "runs/smoke_media_steer/${m#*/}/media_steer.csv" ] || uv run experiments/media_steer.py --stage smoke --model "$m"
    done
    exit 0
fi
for m in $MODELS; do
    [ -f "runs/$RUN/${m#*/}/gate.csv" ] || uv run experiments/media_steer.py --stage gate --model "$m" --run "$RUN"
done
for m in $MODELS; do
    [ -f "runs/$RUN/${m#*/}/media_steer.csv" ] || uv run experiments/media_steer.py --stage main --model "$m" --run "$RUN" --gate-models "$LIST"
done
