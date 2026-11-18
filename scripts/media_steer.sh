#!/usr/bin/env bash
# plan.md, steps 2 to 7, on one GPU machine (a RunPod pod). A step whose output exists is skipped, so the chain picks up where
# it stopped.
#
#   scripts/media_steer.sh                                      steps 5 and 6 on the two medium models (Qwen3.5-4B, Gemma-4-E4B): 48 GB is enough
#   STAGE=smoke scripts/media_steer.sh                          step 5 only: the smoke check twice
#   MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B Qwen/Qwen3.5-27B google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-31B-it" \
#       scripts/media_steer.sh                                  all six (the 27B and the 31B need the 96 GB card)
#
# Cold start: only the first model and the clips are downloaded before the GPU starts (in parallel); every later model is
# downloaded in the background, one after another, while the earlier models run. List the models smallest first, so the GPU
# waits only for the smallest download. Each finished download leaves runs/.downloads/<model>.ok; a model's first step waits
# for its file, and the chain stops if the background download failed (its log: runs/.downloads/<model>.log).
#
# Order: per model (as each arrives) the smoke check twice and the gate; then the comparison of the two smoke runs; then the
# 64-clip run of every model. The 64-clip runs come last because their clips are the ones every model in the run names
# (plan.md), so every gate must be in first; the downloads are over by then.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
RUN=${RUN:-media_steer}
STAGE=${STAGE:-main}
CANDIDATES=${CANDIDATES:-16}   # 32 when a gate leaves a class short (plan.md, "What can still go wrong")
LIST=$(echo "$MODELS" | tr ' ' ,)
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen   # once, so the parallel uv runs below do not each check the environment
if ! uv run --frozen scripts/pin_media.py --check media_steer; then
    echo "pin the datasets and Gemma-4-E2B first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
    exit 1
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

# the first model and the clips, in parallel: this is all the GPU waits for
download "${M[0]}" &
FIRST=$!
uv run --frozen --with av==15.1.0 scripts/prepare_media.py --candidates "$CANDIDATES"   # the clips, decoded once, checked against the committed manifest
wait "$FIRST" || { echo "the download of ${M[0]} failed: see $DL/${M[0]#*/}.log" >&2; exit 1; }

# every later model in the background, one at a time, so each finishes as early as it can
if [ "${#M[@]}" -gt 1 ]; then
    (for m in "${M[@]:1}"; do download "$m" || exit 1; done) &
    BG=$!
fi

# step 5 and the gate, per model as it arrives
for m in "${M[@]}"; do
    wait_for "$m"
    for r in smoke_media_steer smoke_media_steer_repeat; do
        [ -f "runs/$r/${m#*/}/media_steer.csv" ] || uv run --frozen experiments/media_steer.py --stage smoke --model "$m" --run "$r"
    done
    [ "$STAGE" = smoke ] || [ -f "runs/$RUN/${m#*/}/gate.csv" ] || uv run --frozen experiments/media_steer.py --stage gate --model "$m" --run "$RUN"
done
uv run --frozen scripts/compare_media.py runs/smoke_media_steer runs/smoke_media_steer_repeat   # stops the chain if the runs differ
echo "read runs/smoke_media_steer/*/media_steer.csv row by row before the 64-clip run (plan.md step 5)"
[ "$STAGE" = smoke ] && exit 0

# step 6: the 64-clip run of every model
for m in "${M[@]}"; do
    [ -f "runs/$RUN/${m#*/}/media_steer.csv" ] || uv run --frozen experiments/media_steer.py --stage main --model "$m" --run "$RUN" --gate-models "$LIST"
done
