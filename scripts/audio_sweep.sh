#!/usr/bin/env bash
# The audio experiments on one GPU machine (a RunPod pod), in steps that each must pass before the next:
#   1. experiments/audio_clean.py: every model's clean answers to the five questions (receiver and donor recordings, both
#      prompt orders, silence, noise and typed controls), with the sweep's decoding. With SWEEP=audio_sweep the model stops
#      unless ready (both orders); with SWEEP=audio_sweep_qfirst the sweep itself checks the question-first gates.
#      Then experiments/audio_rules.py (RULES=1, the default): the three wordings of the 7-or-3 rules, a diagnosis only.
#   2. a 5-scene patch-only pilot of experiments/audio_sweep.py (clean, self, full donor, per-window arms), twice, compared
#      answer by answer; then its controls must pass (scripts/audio_gate.py pilot). Matching answers alone are not enough.
#      A frozen or typed input the model does not perceive is untestable, as audio_sweep_static's rule says, and goes on.
#   3. the full sweep, only with FULL=1, after the pilot's outputs have been inspected (the CSV and summary.json).
# Before step 1, a model whose lens is not Neuronpedia's (Gemma-4-12B: hlampes/jacobian-lens) must pass scripts/check_lens.py
# against LENS_REF's lens (a Qwen lens fitted here: scripts/validate_lens.py's verdict, config.LENS_CHECKS); otherwise the model is skipped. A step whose output exists is skipped, so the chain picks up where it
# stopped.
#
#   scripts/audio_sweep.sh                                 steps 1 and 2 for Gemma-4-E2B, E4B and 12B (an L40S); then stop
#   FULL=1 scripts/audio_sweep.sh                          step 3 too, for each model whose clean and pilot gates pass
#   SWEEP=audio_sweep_qfirst scripts/audio_sweep.sh        the question-first sweep (runs as audio_sweep_qfirst*)
#   SWEEP=audio_sweep_static scripts/audio_sweep.sh        spoken, frozen and typed words, question first (audio_sweep_static*)
#   SWEEP=audio_sweep_static MODELS="Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B" scripts/audio_sweep.sh     the Qwen omni models
#     (an L40S; Qwen/Qwen3-Omni-30B-A3B-Instruct needs an H200: its text-generating part alone is about 60 GB). Their lenses are
#     in a private Hugging Face repo: the pod's .env needs a token that can read it.
#
# A model whose gate fails is skipped; KEEP_GOING=1 goes on to the next model instead of stopping. DONOR_GB as in
# video_sweep.sh: it must stay the same for a model's pilot and its repeat.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-12B-it"}
LENS_REF=${LENS_REF:-google/gemma-4-E4B-it}   # the model whose Neuronpedia lens sets the bar for a lens from elsewhere
# Gemma-4-12B's other lens (25 of its layers, 1000 prompts), read beside its own at the layers they share: reported, not judged.
# Only the 12B has one; the Qwen omni models' lenses are this project's own fits (scripts/fit_lens.py).
EXTRA_LENS_12B=${EXTRA_LENS_12B:-lamm-mit/gemma4-jacobian-lenses:gemma4-12b-it/paper/seed0.pt@c095b094c52dd7afeb1310bb97ea35dc5005e12f}
CLEAN=${CLEAN:-audio_clean}
SWEEP=${SWEEP:-audio_sweep}   # audio_sweep (both orders), audio_sweep_qfirst (the question first) or audio_sweep_static (with
                              # the frozen and typed counterparts of video_sweep2's frozen video and still picture)
RUN=${RUN:-$SWEEP}
RULES=${RULES:-1}
case "$SWEEP" in audio_sweep|audio_sweep_qfirst|audio_sweep_static) ;; *) echo "SWEEP must be audio_sweep, audio_sweep_qfirst or audio_sweep_static" >&2; exit 1;; esac
DONOR_GB=${DONOR_GB:-16}
KEEP_GOING=${KEEP_GOING:-0}
FULL=${FULL:-0}

stop() {  # stop <message>: the end of this model's chain; the next model only with KEEP_GOING=1
    if [ "$KEEP_GOING" = 1 ]; then echo "audio_sweep: $1; next model (KEEP_GOING=1)" >&2; else echo "audio_sweep: $1" >&2; exit 1; fi
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

model() {  # model <model>: the three steps; returns non-zero at the first step that fails
    local m=$1 tag=${1#*/}
    wait_for "$m"
    # 0. a lens from outside Neuronpedia: checked on the model against the reference model's lens first
    # a Qwen omni model runs on its 1000-prompt lens only: decided before any check_lens run (config.DECIDING_LENS)
    uv run --frozen python -c "import sys; from jspace.config import deciding_lens_pinned; sys.exit(0 if deciding_lens_pinned(sys.argv[1]) else 1)" "$m" ||
        { echo "$m: the pinned lens is not its deciding 1000-prompt lens; pin that first (docs/omni_audio_run.md)" >&2; return 1; }
    if [ "${m%%/*}" = Qwen ] && uv run --frozen python -c "import sys; from jspace.config import MODELS; sys.exit(0 if MODELS[sys.argv[1]].lens_repo else 1)" "$m"; then
        # a Qwen lens fitted here: used only if scripts/validate_lens.py passed it (config.LENS_CHECKS), and only the lens it passed
        uv run --frozen python -c "import json, sys; from jspace.config import MODELS; s = json.load(open(sys.argv[1])); print({k: s[k]['passes'] for k in ('reproduce', 'stable', 'reads_concepts')}); sys.exit(0 if s['passes'] and s['lens']['sha256'] == MODELS[sys.argv[2]].lens_sha256 else 1)" "runs/lens_validation/$tag/summary.json" "$m" ||
            { echo "$m: its pinned lens has no passing verdict in runs/lens_validation (docs/lens_validation_run.md)" >&2; return 1; }
    elif uv run --frozen python -c "import sys; from jspace.config import MODELS; sys.exit(0 if MODELS[sys.argv[1]].lens_repo else 1)" "$m"; then
        download "$LENS_REF" || { echo "the download of $LENS_REF failed" >&2; return 1; }
        [ -f "runs/check_lens/${LENS_REF#*/}/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$LENS_REF" --run check_lens || { echo "check_lens of $LENS_REF failed" >&2; return 1; }
        local extra=()
        [ "$m" = google/gemma-4-12B-it ] && extra=(--extra-lens "$EXTRA_LENS_12B")
        [ -f "runs/check_lens/$tag/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$m" --run check_lens --reference "$LENS_REF" ${extra[@]+"${extra[@]}"} || { echo "check_lens of $m failed" >&2; return 1; }
        uv run --frozen scripts/audio_gate.py lens "runs/check_lens/$tag/summary.json" || { echo "$m: its lens fails the check" >&2; return 1; }
    fi
    # 1. clean answers, then their gates
    [ -f "runs/$CLEAN/$tag/audio_clean.csv" ] || uv run --frozen experiments/audio_clean.py --model "$m" --run "$CLEAN" || { echo "the $CLEAN run of $m failed" >&2; return 1; }
    if [ "$RULES" = 1 ]; then  # a diagnosis, never a gate
        [ -f "runs/audio_rules/$tag/audio_rules.csv" ] || uv run --frozen experiments/audio_rules.py --model "$m" --run audio_rules --clean-run "$CLEAN" || { echo "the audio_rules run of $m failed" >&2; return 1; }
    fi
    if [ "$SWEEP" = audio_sweep ]; then  # the question-first sweeps check their own one-order gates before loading the model
        uv run --frozen scripts/audio_gate.py clean "runs/$CLEAN/$tag/summary.json" || { echo "$m: the clean answers fail their gates" >&2; return 1; }
    fi
    # 2. the patch-only pilot twice, the same answers both times, and its controls
    for r in "${RUN}_pilot" "${RUN}_pilot_repeat"; do
        [ -f "runs/$r/$tag/audio_sweep.csv" ] || uv run --frozen "experiments/$SWEEP.py" --model "$m" --run "$r" --gate-run "$CLEAN" --parts patch --n 5 --donor-gb "$DONOR_GB" || { echo "the $r run of $m failed" >&2; return 1; }
    done
    # this model's pair only: on a shared volume another pod's model may still be mid-repeat
    uv run --frozen scripts/compare_media.py --file audio_sweep.csv --model "$tag" "runs/${RUN}_pilot" "runs/${RUN}_pilot_repeat" || { echo "$m: the two pilot runs differ (listed above)" >&2; return 1; }
    uv run --frozen scripts/audio_gate.py pilot "runs/${RUN}_pilot/$tag/summary.json" || { echo "$m: the pilot's controls fail" >&2; return 1; }
    # 3. the full sweep, only when asked for after inspection
    if [ "$FULL" != 1 ]; then
        echo "$m: clean and pilot gates pass. Inspect runs/$CLEAN/$tag/audio_clean.csv and runs/${RUN}_pilot/$tag/ (audio_sweep.csv, summary.json), then run SWEEP=$SWEEP FULL=1 $0"
        return 0
    fi
    [ -f "runs/$RUN/$tag/audio_sweep.csv" ] || uv run --frozen "experiments/$SWEEP.py" --model "$m" --run "$RUN" --gate-run "$CLEAN" --donor-gb "$DONOR_GB" || { echo "the $RUN run of $m failed" >&2; return 1; }
}

for m in "${M[@]}"; do
    model "$m" || stop "$m stopped"
done
