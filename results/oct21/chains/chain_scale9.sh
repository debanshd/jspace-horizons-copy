#!/bin/bash
# R23 (matched-disruption source test, 4B), R24 (channel-first order, 4B), R25 (answer-held source test on Gemma-4-E4B by knockout);
# one after another, beside the 9B knockout (chain 8); rules in RECORD.md (21 Oct 14:39 local)
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
$PY experiments/source_cause.py --model Qwen/Qwen3.5-4B --run run20 > runs/source_cause_4B.log 2>&1
log "R23 source_cause 4B exit $?"
$PY experiments/source_conflict.py --model Qwen/Qwen3.5-4B --run run21 --congruent --order channel-first > runs/source_order_4B.log 2>&1
log "R24 channel-first 4B exit $?"
$PY experiments/source_conflict.py --model google/gemma-4-E4B-it --run run22 --congruent > runs/source_conflict_E4B.log 2>&1
log "R25 conflict E4B exit $?"
log "chain 9 done"
