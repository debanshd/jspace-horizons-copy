#!/bin/bash
# the adversarial council's three runs (R17, R18, R19), after the conflict runs, beside the 27B
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
PY=/workspace/nayan/jspace-horizons/.venv/bin/python
until grep -q "4B exit" runs/conflict_now.log 2>/dev/null; do sleep 20; done
$PY experiments/cross_modal_patch.py --model Qwen/Qwen3.5-2B --run run13m --questions mc --layers 7,11,15,19 --joint "7+11,11+15+19" --directions "typed>picture,picture>picture" > runs/crossmodal_mc_2B.log 2>&1; echo "R17 exit $?"
$PY experiments/source_report.py --model Qwen/Qwen3.5-4B --run run15c --donor-control --layers "19+23,15+19+23,11+15+19+23" > runs/source_control_4B.log 2>&1; echo "R18 exit $?"
$PY experiments/cross_modal_patch.py --model Qwen/Qwen3.5-2B --run run13r --crop-set 0 --layers 7,11,15,19 --joint "7+11,11+15+19" --directions "typed>picture,picture>picture" > runs/crossmodal_rep_2B.log 2>&1; echo "R19 exit $?"
echo "council runs done"
