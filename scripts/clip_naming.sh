#!/usr/bin/env bash
# experiments/clip_naming.py on one GPU machine (a RunPod pod): each model names the 73 matched clips, shown whole, with the
# COCO-category question. A step whose output exists is skipped, so the chain picks up where it stopped.
#
#   scripts/clip_naming.sh                                   the two small and the two medium models (a 48 GB card is enough)
#   MODELS="Qwen/Qwen3.5-27B google/gemma-4-31B-it" scripts/clip_naming.sh     the large ones (96 GB card)
#
# PyAV decodes the clips (`uv run --with av`), so uv.lock is untouched. Later models download while earlier ones run.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
RUN=${RUN:-clip_naming}
AV="--with av==15.1.0"
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen
if ! uv run --frozen scripts/pin_media.py --check video_sweep; then  # clip_naming reads the same matched set as video_sweep
    echo "pin the matched video set first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
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
    [ -f "runs/$RUN/${m#*/}/clip_naming.csv" ] || uv run --frozen $AV experiments/clip_naming.py --model "$m" --run "$RUN"
done
