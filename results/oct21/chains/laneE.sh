#!/usr/bin/env bash
# the batteries and the E4B knockout, after the volume quota incident; the 0.8B knockout runs beside the first of these
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-4B --run run7 --questions all --step 1 --mix > runs/run7/battery_4B.log 2>&1
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-2B --run run7 --questions all --step 1 --mix > runs/run7/battery_2B.log 2>&1
uv run --offline experiments/attention_knockout.py --model google/gemma-4-E4B-it --run run7k --questions say,letter,category,set,animal > runs/run7k/knockout_E4B.log 2>&1
uv run --offline experiments/read_battery.py --model google/gemma-4-E4B-it --run run7 --questions 2x2 --step 2 > runs/run7/battery_E4B.log 2>&1
echo "lane E done $(date)" > runs/laneE.done
