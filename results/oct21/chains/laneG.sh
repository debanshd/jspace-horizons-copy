#!/usr/bin/env bash
cd /workspace/nayan/jspace-horizons
export HF_HOME=/workspace/hf
set -a; . /workspace/nayan/.env; set +a
while [ ! -f runs/laneF.done ]; do sleep 30; done
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-2B --run run7t --questions all --typed > runs/run7k/typed_2B.log 2>&1
uv run --offline experiments/attention_knockout.py --model Qwen/Qwen3.5-4B --run run7t --questions all --typed > runs/run7k/typed_4B.log 2>&1
echo "lane G done $(date)" > runs/laneG.done
