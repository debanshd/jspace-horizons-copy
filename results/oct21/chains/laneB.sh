#!/usr/bin/env bash
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-2B --run run7 --questions all --step 1 --mix > runs/run7/battery_2B.log 2>&1
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-2B --run run7k --questions all > runs/run7k/knockout_2B.log 2>&1
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-0.8B --run run7 --questions all --step 1 --mix > runs/run7/battery_08B.log 2>&1
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-0.8B --run run7k --questions all > runs/run7k/knockout_08B.log 2>&1
echo "lane B done" > runs/laneB.done
