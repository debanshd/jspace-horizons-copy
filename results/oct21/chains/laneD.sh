#!/usr/bin/env bash
# the batteries, strictly one GPU job at a time (the earlier deaths were log writes failing on a full volume quota)
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
while pgrep -f "experiments/(read_battery|attention_knockout).py" > /dev/null; do sleep 20; done
for M in Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B Qwen/Qwen3.5-0.8B; do
  T=${M#*/}
  uv run --offline experiments/read_battery.py --model $M --run run7 --questions all --step 1 --mix > runs/run7/battery_$T.log 2>&1
done
uv run --offline experiments/attention_knockout.py --model google/gemma-4-E4B-it --run run7k --questions all > runs/run7k/knockout_E4B.log 2>&1
echo "lane D done $(date)" > runs/laneD.done
