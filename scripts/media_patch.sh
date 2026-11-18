#!/usr/bin/env bash
# plan.md part 2 on one GPU machine (a RunPod pod): video patching, depth_sweep's design, on the VGGSound clips of the six-model
# steering run, as the clip and its static version. A step whose output exists is skipped, so the chain picks up where it stopped.
#
#   scripts/media_patch.sh                                  the six models of the steering run, smallest first (96 GB card)
#   MODELS="Qwen/Qwen3.5-4B google/gemma-4-E4B-it" scripts/media_patch.sh      two models (48 GB is enough)
#
# The clips are those every model of runs/media_steer_all named; on a fresh pod their gate.csv files are taken from the
# committed archive docs/media_steer_runs.tar.gz. Later models download in the background while earlier ones run.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it Qwen/Qwen3.5-27B google/gemma-4-31B-it"}
RUN=${RUN:-media_patch}
GATE_RUN=${GATE_RUN:-media_steer_all}
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen
if ! uv run --frozen scripts/pin_media.py --check media_patch; then
    echo "pin the datasets first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
    exit 1
fi
if ! ls runs/"$GATE_RUN"/*/gate.csv > /dev/null 2>&1; then
    echo "runs/$GATE_RUN has no gate.csv: taking the gate files from docs/media_steer_runs.tar.gz"
    tar -xzf docs/media_steer_runs.tar.gz --wildcards "runs/$GATE_RUN/*/gate.csv"
fi

download() {  # download <model>: the model and its lens, checked; leaves the .ok file on success
    local m=$1 tag=${1#*/}
    [ -f "$DL/$tag.ok" ] && return 0
    uv run --frozen scripts/download.py --skip-data --models "$m" > "$DL/$tag.log" 2>&1 && touch "$DL/$tag.ok"
}

wait_for() {  # wait_for <model>: until its background download is done; stop if it failed
    local tag=${1#*/}
    while [ ! -f "$DL/$tag.ok" ]; do
        if [ -n "${BG:-}" ] && ! kill -0 "$BG" 2>/dev/null && [ ! -f "$DL/$tag.ok" ]; then
            echo "the download of $1 failed: see $DL/$tag.log" >&2
            exit 1
        fi
        sleep 10
    done
}

# the first model and the VGGSound clips (checked against their committed manifest), in parallel
download "${M[0]}" &
FIRST=$!
uv run --frozen --with av==15.1.0 scripts/prepare_media.py --candidates 32
wait "$FIRST" || { echo "the download of ${M[0]} failed: see $DL/${M[0]#*/}.log" >&2; exit 1; }
if [ "${#M[@]}" -gt 1 ]; then
    (for m in "${M[@]:1}"; do download "$m" || exit 1; done) &
    BG=$!
fi

for m in "${M[@]}"; do
    wait_for "$m"
    [ -f "runs/$RUN/${m#*/}/media_patch.csv" ] || uv run --frozen experiments/media_patch.py --model "$m" --run "$RUN" --gate-run "$GATE_RUN"
done
