#!/usr/bin/env bash
# experiments/video_sweep2.py on one GPU machine (a RunPod pod): video_sweep's patching and interventions on each model's gated
# schedule of the matched clips, shown whole. A step whose output exists is skipped, so the chain picks up where it stopped.
#
#   scripts/video_sweep2.sh                                  the two small and the two medium models (a 48 GB card is enough)
#   MODELS="Qwen/Qwen3.5-27B google/gemma-4-31B-it" scripts/video_sweep2.sh    the large ones, once their naming gate has run
#
# The gate: each Qwen model reads runs/clip_naming/<model>/summary.json and each Gemma model runs/clip_naming2/<model>/summary.json
# (PR #7). On a fresh pod those summaries come from the committed archives docs/clip_naming_runs.tar.gz and
# docs/clip_naming2_runs.tar.gz; a model with no gate file stops the chain.
# Per model: the 5-scene smoke check twice, compared answer by answer (the chain stops if they differ, unless KEEP_GOING=1),
# then the full gated schedule. DONOR_GB as in video_sweep.sh: it must stay the same for a model's smoke run and its repeat.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it"}
RUN=${RUN:-video_sweep2}
DONOR_GB=${DONOR_GB:-16}
KEEP_GOING=${KEEP_GOING:-0}

keep_going() {
    if [ "$KEEP_GOING" = 1 ]; then echo "video_sweep2: $1; going on (KEEP_GOING=1)" >&2; else echo "video_sweep2: $1" >&2; exit 1; fi
}
gate_of() {  # gate_of <model>: the naming run whose gated schedule the model runs on
    case "$1" in google/gemma-*) echo clip_naming2 ;; *) echo clip_naming ;; esac
}
AV="--with av==15.1.0"
read -r -a M <<< "$MODELS"
DL=runs/.downloads
mkdir -p "$DL"

uv sync --frozen
if ! uv run --frozen scripts/pin_media.py --check video_sweep; then  # video_sweep2 reads the same matched set as video_sweep
    echo "pin the matched video set first: uv run scripts/pin_media.py, then commit src/jspace/config.py" >&2
    exit 1
fi
for m in "${M[@]}"; do
    g=$(gate_of "$m")
    f="runs/$g/${m#*/}/summary.json"
    if [ ! -f "$f" ] && [ -f "docs/${g}_runs.tar.gz" ]; then
        tar -xzf "docs/${g}_runs.tar.gz" "$f" 2>/dev/null || true
    fi
    [ -f "$f" ] || { echo "no naming gate for $m: $f (run scripts/$g.sh for it first)" >&2; exit 1; }
done

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
    g=$(gate_of "$m")
    for r in "${RUN}_smoke" "${RUN}_smoke_repeat"; do
        [ -f "runs/$r/${m#*/}/video_sweep2.csv" ] || uv run --frozen $AV experiments/video_sweep2.py --model "$m" --run "$r" --gate-run "$g" --n 5 --donor-gb "$DONOR_GB" || keep_going "the $r run of $m failed"
    done
    uv run --frozen scripts/compare_media.py --file video_sweep2.csv "runs/${RUN}_smoke" "runs/${RUN}_smoke_repeat" || keep_going "the two smoke runs of $m differ (listed above)"
    [ -f "runs/$RUN/${m#*/}/video_sweep2.csv" ] || uv run --frozen $AV experiments/video_sweep2.py --model "$m" --run "$RUN" --gate-run "$g" --donor-gb "$DONOR_GB" || keep_going "the $RUN run of $m failed"
done
