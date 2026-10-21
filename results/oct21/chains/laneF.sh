#!/usr/bin/env bash
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
while pgrep -f "read_battery.py --model Qwen/Qwen3.5-4B" > /dev/null; do sleep 30; done
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-2B --run run7h --questions say,animal,set,category --heads 7,11,15,19 > runs/run7k/heads_2B.log 2>&1
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-4B --run run7h --questions say,animal,set,category --heads 7,11,19,23 > runs/run7k/heads_4B.log 2>&1
echo "lane F done $(date)" > runs/laneF.done
