#!/usr/bin/env bash
# plan.md part 3 on one GPU machine (a RunPod pod): depth_sweep's patching and media_steer's interventions on the matched video
# set (saxenan3/jspace-matched-video), with each model's own frame sampler. A step whose output exists is skipped.
#
#   scripts/video_sweep.sh                                   the two small and the two medium models (a 48 GB card is enough)
#   MODELS="Qwen/Qwen3.5-27B google/gemma-4-31B-it" scripts/video_sweep.sh     the large ones (96 GB card; several hours each)
#
# Per model: the 5-scene smoke check twice, compared answer by answer (the chain stops if they differ), then the 64-scene run.
# PyAV decodes the clips (`uv run --with av`), so uv.lock is untouched. Later models download while earlier ones run.
# DONOR_GB is the host memory for donor states (--donor-gb, default 16): scenes run in chunks that fit it, and only scenes of one
# chunk share batches, so a larger value gives fuller batches. It must stay the same for a model's smoke run and its repeat.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
RUN=${RUN:-video_sweep}
DONOR_GB=${DONOR_GB:-16}
AV="--with av==15.1.0"
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen
if ! uv run --frozen scripts/pin_media.py --check video_sweep; then
    echo "pin the matched video set first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
    exit 1
fi

download() {
    local m=$1 tag=${1#*/}
    [ -f "$DL/$tag.ok" ] && return 0
    uv run --frozen scripts/download.py --skip-data --models "$m" > "$DL/$tag.log" 2>&1 && touch "$DL/$tag.ok"
}

wait_for() {
    local tag=${1#*/}
    while [ ! -f "$DL/$tag.ok" ]; do
        if [ -n "${BG:-}" ] && ! kill -0 "$BG" 2>/dev/null && [ ! -f "$DL/$tag.ok" ]; then
            echo "the download of $1 failed: see $DL/$tag.log" >&2
            exit 1
        fi
        sleep 10
    done
}

download "${M[0]}" &
FIRST=$!
uv run --frozen scripts/prepare_matched.py   # the clips, each checked against the sha256 in the dataset's ATTRIBUTION.tsv
wait "$FIRST" || { echo "the download of ${M[0]} failed: see $DL/${M[0]#*/}.log" >&2; exit 1; }
if [ "${#M[@]}" -gt 1 ]; then
    (for m in "${M[@]:1}"; do download "$m" || exit 1; done) &
    BG=$!
fi

for m in "${M[@]}"; do
    wait_for "$m"
    for r in "${RUN}_smoke" "${RUN}_smoke_repeat"; do
        [ -f "runs/$r/${m#*/}/video_sweep.csv" ] || uv run --frozen $AV experiments/video_sweep.py --model "$m" --run "$r" --n 5 --donor-gb "$DONOR_GB"
    done
    uv run --frozen scripts/compare_media.py --file video_sweep.csv "runs/${RUN}_smoke" "runs/${RUN}_smoke_repeat"
    [ -f "runs/$RUN/${m#*/}/video_sweep.csv" ] || uv run --frozen $AV experiments/video_sweep.py --model "$m" --run "$RUN" --donor-gb "$DONOR_GB"
done
