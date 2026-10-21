#!/usr/bin/env bash
# the 4B and 2B batteries again (their first processes died silently at 03:19 while two smoke runs loaded), after the knockouts
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
while pgrep -f "attention_knockout.py --model Qwen/Qwen3.5-[42]B" > /dev/null; do sleep 20; done
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-4B --run run7 --questions all --step 1 --mix > runs/run7/battery_4B.log 2>&1 &
P1=$!
uv run --offline experiments/read_battery.py --model Qwen/Qwen3.5-2B --run run7 --questions all --step 1 --mix > runs/run7/battery_2B.log 2>&1 &
P2=$!
wait $P1 $P2
echo "lane C done $(date)" > runs/laneC.done
