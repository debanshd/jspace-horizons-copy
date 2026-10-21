#!/bin/bash
# replaces chain 6: the quota test, the 9B download and knockout, the 31B with every layer blockable (R13c), then the E4B
# interchange and source report (R16); one model at a time on the card
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
until ! pgrep -x dd > /dev/null 2>&1; do sleep 5; done
rm -f /workspace/hf/.quota_test
if dd if=/dev/zero of=/workspace/hf/.quota_test bs=1M count=25000 status=none; then
  rm -f /workspace/hf/.quota_test; log "quota ok for 25 GB"
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))" > runs/download_9B.log 2>&1 && log "9B downloaded" && $PY experiments/attention_knockout.py --model Qwen/Qwen3.5-9B --run run7k --questions say,letter,category,set,animal,mc > runs/knockout_9B.log 2>&1
  log "9B knockout exit $?"
else
  rm -f /workspace/hf/.quota_test; log "QUOTA TEST FAILED, 9B skipped"
fi
$PY experiments/attention_knockout.py --model google/gemma-4-31B-it --run run7ka --questions say,animal > runs/knockout_31B_all.log 2>&1
log "31B every-layer knockout exit $?"
$PY experiments/cross_modal_patch.py --model google/gemma-4-E4B-it --run run13 --layers 11,17,29 --joint "11+17,17+29,11+17+29" --directions "typed>picture,picture>picture" > runs/crossmodal_E4B.log 2>&1
log "E4B cross-modal exit $?"
$PY experiments/source_report.py --model google/gemma-4-E4B-it --run run15 --layers "17+29,11+17+29" > runs/source_E4B.log 2>&1
log "E4B source report exit $?"
log "chain 7 done"
