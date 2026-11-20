#!/usr/bin/env bash
# The audio experiments on one GPU machine (a RunPod pod): experiments/audio_naming.py (which spoken recordings each model
# names, in both prompt orders, with the silence, noise and typed controls), then experiments/audio_sweep.py (video_sweep2's
# patching and interventions on the scenes whose two recordings the model names). A step whose output exists is skipped, so
# the chain picks up where it stopped.
#
#   scripts/audio_sweep.sh                                         Gemma-4-E2B and E4B, the audio-capable models (an L40S)
#
# Per model: audio_naming, then the 5-scene smoke check of audio_sweep twice, compared answer by answer (the chain stops if
# they differ, unless KEEP_GOING=1), then the full gated schedule. DONOR_GB as in video_sweep.sh: it must stay the same for a
# model's smoke run and its repeat.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"google/gemma-4-E2B-it google/gemma-4-E4B-it"}
NAMING=${NAMING:-audio_naming}
RUN=${RUN:-audio_sweep}
DONOR_GB=${DONOR_GB:-16}
KEEP_GOING=${KEEP_GOING:-0}

keep_going() {
    if [ "$KEEP_GOING" = 1 ]; then echo "audio_sweep: $1; going on (KEEP_GOING=1)" >&2; else echo "audio_sweep: $1" >&2; exit 1; fi
}
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen

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

download "${M[0]}" &
FIRST=$!
uv run --frozen scripts/prepare_spoken.py   # the recordings, each checked against the dataset's checksums.json
wait "$FIRST" || { echo "the download of ${M[0]} failed: see $DL/${M[0]#*/}.log" >&2; exit 1; }
if [ "${#M[@]}" -gt 1 ]; then
    (for m in "${M[@]:1}"; do download "$m" || exit 1; done) &
    BG=$!
fi

for m in "${M[@]}"; do
    wait_for "$m"
    [ -f "runs/$NAMING/${m#*/}/audio_naming.csv" ] || uv run --frozen experiments/audio_naming.py --model "$m" --run "$NAMING" || keep_going "the $NAMING run of $m failed"
    for r in "${RUN}_smoke" "${RUN}_smoke_repeat"; do
        [ -f "runs/$r/${m#*/}/audio_sweep.csv" ] || uv run --frozen experiments/audio_sweep.py --model "$m" --run "$r" --gate-run "$NAMING" --n 5 --donor-gb "$DONOR_GB" || keep_going "the $r run of $m failed"
    done
    # this model's pair only: on a shared volume another pod's model may still be mid-repeat
    uv run --frozen scripts/compare_media.py --file audio_sweep.csv --model "${m#*/}" "runs/${RUN}_smoke" "runs/${RUN}_smoke_repeat" || keep_going "the two smoke runs of $m differ (listed above)"
    [ -f "runs/$RUN/${m#*/}/audio_sweep.csv" ] || uv run --frozen experiments/audio_sweep.py --model "$m" --run "$RUN" --gate-run "$NAMING" --donor-gb "$DONOR_GB" || keep_going "the $RUN run of $m failed"
done
