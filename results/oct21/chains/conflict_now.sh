#!/bin/bash
# the two-source conflict on the 2B and 4B beside the 27B knockout (8 and 12 GB), so R14 lands before the chain reaches it
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
$PY experiments/source_conflict.py --model Qwen/Qwen3.5-2B --run run17 > runs/conflict_2B.log 2>&1; echo "2B exit $?"
$PY experiments/source_conflict.py --model Qwen/Qwen3.5-4B --run run17 > runs/conflict_4B.log 2>&1; echo "4B exit $?"
