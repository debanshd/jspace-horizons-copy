#!/bin/bash
# the size step, one model at a time on the card: 27B knockout, its stations by rule, 27B interchange, source report and head-level
# knockout, Gemma-4-31B knockout (weights staged in /dev/shm), the 9B after a quota test, then the E4B battery
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
until [ -f runs/run7k/gemma-4-E4B-it/summary.json ]; do sleep 30; done
log "E4B knockout done"
$PY experiments/attention_knockout.py --model Qwen/Qwen3.5-27B --run run7k --questions say,letter,category,set,animal,mc > runs/knockout_27B.log 2>&1
log "27B knockout exit $?"
read -r D C <<< "$($PY - <<'PYEOF'
import json
s = json.load(open("runs/run7k/Qwen3.5-27B/summary.json"))["questions"]
AL = json.load(open("runs/run7k/Qwen3.5-27B/summary.json"))["attention_layers"]
def last(t):
    v = s.get(t, {})
    return v.get("last_reader") or max((int(L) for L, d in v.get("single", {}).items() if d["cost"][0] >= 1), default=None)
D = last("animal") or last("category") or AL[2]
C = last("say") or last("letter") or AL[len(AL) * 2 // 3]
print(D, C)
PYEOF
)"
log "27B stations by rule: decision $D, copy $C"
D1=$((D-4)); C1=$((C-4)); C2=$((C-8))
echo "| 27B chain (auto, $(date '+%d %b %H:%M')) | layers for the 27B interchange and source report chosen by the rule of 12:10 from runs/run7k/Qwen3.5-27B: decision station $D, copy station $C; interchange layers $D1,$D,$C2,$C1,$C with joint $D1+$D, $C2+$C1+$C and all five; source report $C1+$C, $C2+$C1+$C, all five |" >> /workspace/nayan/RECORD.md
$PY experiments/cross_modal_patch.py --model Qwen/Qwen3.5-27B --run run13 --layers $D1,$D,$C2,$C1,$C --joint "$D1+$D,$C2+$C1+$C,$D1+$D+$C2+$C1+$C" --directions "typed>picture,picture>picture" > runs/crossmodal_27B.log 2>&1
log "27B cross-modal exit $?"
$PY experiments/source_report.py --model Qwen/Qwen3.5-27B --run run15 --layers "$C1+$C,$C2+$C1+$C,$D1+$D+$C2+$C1+$C" > runs/source_27B.log 2>&1
log "27B source report exit $?"
$PY experiments/attention_knockout.py --model Qwen/Qwen3.5-27B --run run7h --questions animal,set,category,say --heads $D,$C --layers $D,$C > runs/heads_27B.log 2>&1
log "27B heads exit $?"
until grep -q "^exit" /dev/shm/download_31B.log 2>/dev/null; do sleep 30; done
if grep -q "^exit 0" /dev/shm/download_31B.log && [ -d /dev/shm/hf/hub/models--google--gemma-4-31B-it ]; then
  [ -e /workspace/hf/hub/models--google--gemma-4-31B-it ] || ln -s /dev/shm/hf/hub/models--google--gemma-4-31B-it /workspace/hf/hub/models--google--gemma-4-31B-it
  log "31B staged"
  $PY experiments/attention_knockout.py --model google/gemma-4-31B-it --run run7k --questions say,category,animal,mc --layers 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57 > runs/knockout_31B.log 2>&1
  log "31B knockout exit $?"
else
  log "31B download failed, skipped"
fi
if dd if=/dev/zero of=/workspace/hf/.quota_test bs=1M count=25000 status=none; then
  rm -f /workspace/hf/.quota_test; log "quota ok for 25 GB"
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))" > runs/download_9B.log 2>&1 && log "9B downloaded" && $PY experiments/attention_knockout.py --model Qwen/Qwen3.5-9B --run run7k --questions say,letter,category,set,animal,mc > runs/knockout_9B.log 2>&1
  log "9B knockout exit $?"
else
  rm -f /workspace/hf/.quota_test; log "QUOTA TEST FAILED, 9B skipped"
fi
$PY experiments/read_battery.py --model google/gemma-4-E4B-it --run run7 --questions 2x2 --step 2 > runs/run7/battery_E4B.log 2>&1
log "E4B battery exit $?"
log "chain done"
