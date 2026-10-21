#!/bin/bash
# the 9B staged in memory (the volume is at quota), its knockout after the 31B every-layer run; beside the E4B interchange (31 + 25 GB)
cd /workspace/nayan/jspace-horizons
export PYTHONUNBUFFERED=1
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
mkdir -p /dev/shm/hf/hub
HF_HOME=/dev/shm/hf $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a', allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', '*.py']))" > runs/download_9B.log 2>&1 || { log "9B DOWNLOAD FAILED"; exit 1; }
[ -e /workspace/hf/hub/models--Qwen--Qwen3.5-9B ] || ln -s /dev/shm/hf/hub/models--Qwen--Qwen3.5-9B /workspace/hf/hub/models--Qwen--Qwen3.5-9B
log "9B staged in memory"
until [ -f runs/run7ka/gemma-4-31B-it/summary.json ]; do sleep 30; done
export HF_HOME=/workspace/hf
$PY experiments/attention_knockout.py --model Qwen/Qwen3.5-9B --run run7k --questions say,letter,category,set,animal,mc > runs/knockout_9B.log 2>&1
log "9B knockout exit $?"
log "chain 8 done"
