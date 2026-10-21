#!/usr/bin/env bash
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-4B --run run7 --questions all --step 1 --mix > runs/run7/battery_4B.log 2>&1
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-4B --run run7k --questions all > runs/run7k/knockout_4B.log 2>&1
uv run --offline experiments/read_battery.py --model google/gemma-4-E4B-it --run run7 --questions all --step 1 --mix > runs/run7/battery_E4B.log 2>&1
uv run --offline experiments/attention_knockout.py --model google/gemma-4-E4B-it --run run7k --questions all > runs/run7k/knockout_E4B.log 2>&1
echo "lane A done" > runs/laneA.done
