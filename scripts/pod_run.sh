#!/usr/bin/env bash
# This PR's two experiments on one RunPod pod as one chain, and the pod stopped once the chain is over or its GPU sits idle
# (scripts/gpu_watchdog.py), so a finished or stuck run does not keep a paid GPU.
#
#   nohup scripts/pod_run.sh > runs/pod_run.log 2>&1 &         video_sweep, then media_patch, on the six models (96 GB card)
#   MODELS="Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it" scripts/pod_run.sh
#                                                               the four smaller models (a 48 GB card is enough)
#   STAGES=video_sweep scripts/pod_run.sh                       one of the two
#   STOP_POD=0 scripts/pod_run.sh                               without the watchdog
#
# Both scripts skip a step whose output exists, so after a stop the same command picks up where the chain was. The models and
# the VGGSound clips already on the network volume are not downloaded again (runs/.downloads/*.ok).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
export MODELS=${MODELS:-"Qwen/Qwen3.5-0.8B google/gemma-4-E2B-it Qwen/Qwen3.5-4B google/gemma-4-E4B-it Qwen/Qwen3.5-27B google/gemma-4-31B-it"}
STAGES=${STAGES-"video_sweep media_patch"}
STOP_POD=${STOP_POD:-1}
# the owner's choice for this chain: a smoke difference or a failed run is logged, and the chain goes on (KEEP_GOING=0 stops)
export KEEP_GOING=${KEEP_GOING:-1}

# video_sweep's donor states in host memory: about a third of what is free, because the encoded prompts of a chunk (their pixels)
# grow with it too. A chunk shares batches, so the larger it is, the fuller the batches. Set DONOR_GB to fix it across pods.
avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
# in a pod /proc/meminfo shows the whole host (1.4 TB on an L40S machine): the container's cgroup limit is the real one
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
if [[ "$lim" =~ ^[0-9]+$ ]] && [ $((lim / 1073741824)) -lt "$avail" ]; then avail=$((lim / 1073741824)); fi
# nproc shows the host's cores too; torch and the processors spawn a thread per core seen, so cap them at the pod's vCPUs
if [ -n "${RUNPOD_CPU_COUNT:-}" ]; then
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$RUNPOD_CPU_COUNT} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$RUNPOD_CPU_COUNT}
fi
export DONOR_GB=${DONOR_GB:-$(( avail * 35 / 100 > 16 ? avail * 35 / 100 : 16 ))}
echo "pod_run: $(date -u '+%F %T') stages: $STAGES; models: $MODELS; ${avail} GB host memory, DONOR_GB=$DONOR_GB, CPU threads ${OMP_NUM_THREADS:-all}"

if [ "$STOP_POD" = 1 ]; then
    # it watches this shell: when the chain exits, finished or failed, or the GPU idles for 20 minutes, it stops the pod
    python3 scripts/gpu_watchdog.py --pid $$ >> runs/gpu_watchdog.log 2>&1 &
    echo "pod_run: watchdog pid $! (runs/gpu_watchdog.log)"
fi

for s in $STAGES; do
    echo "pod_run: $(date -u '+%F %T') ===== $s"
    "scripts/$s.sh" || { [ "$KEEP_GOING" = 1 ] && echo "pod_run: $s failed; going on to the next stage" >&2; } || exit 1
done
echo "pod_run: $(date -u '+%F %T') all done"
