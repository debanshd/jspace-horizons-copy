#!/bin/bash
# after chain_scale2: the interchange and the source report on Gemma-4-E4B (R16)
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
until grep -q "chain done" runs/chain_scale2.log 2>/dev/null; do sleep 60; done
log "chain 2 done"
$PY experiments/cross_modal_patch.py --model google/gemma-4-E4B-it --run run13 --layers 11,17,29 --joint "11+17,17+29,11+17+29" --directions "typed>picture,picture>picture" > runs/crossmodal_E4B.log 2>&1
log "E4B cross-modal exit $?"
$PY experiments/source_report.py --model google/gemma-4-E4B-it --run run15 --layers "17+29,11+17+29" > runs/source_E4B.log 2>&1
log "E4B source report exit $?"
log "chain 3 done"
