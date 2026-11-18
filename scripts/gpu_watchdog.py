#!/usr/bin/env python
"""Stop the RunPod pod this runs on when its work is over: when the chain it watches exits, or when the GPU has been idle for
too long once it has been busy. Pod time is paid by the hour; an idle pod after a finished or stuck run is pure cost.

    python scripts/gpu_watchdog.py --pid <chain pid>                    stop the pod when the chain ends or the GPU idles
    python scripts/gpu_watchdog.py --pid <chain pid> --dry-run          print what it would do, stop nothing

Rules, in order, checked every --every seconds:
1. The chain (--pid) has exited: stop the pod. Its outputs and logs are on the network volume, which a stop keeps.
2. The GPU has been above --arm-above % at least once (so the long CPU-only start, downloads and clip decoding, never counts),
   and every sample in the last --idle-minutes is below --idle-below %: stop the pod. The gaps between runs (a model loading
   from the volume, prompts encoded on the CPU) last minutes, so the default window of 20 minutes is several times longer.

The stop is `runpodctl stop pod $RUNPOD_POD_ID`, which RunPod puts in every pod with a pod-scoped key; if that fails, the REST
v2 call `POST /v2/pods/{id}/action {"action": "stop"}` with $RUNPOD_API_KEY. A stop keeps the volume and the container disk.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def decide(samples, now, *, chain_alive, armed, idle_minutes, idle_below):
    """'stop: <why>' or None, from [(time, utilization %)] samples, oldest first; armed: the GPU has been busy once"""
    if not chain_alive:
        return "stop: the chain has exited"
    if not armed:
        return None  # the GPU has not worked yet
    window = [u for t, u in samples if t >= now - idle_minutes * 60]
    covered = samples and samples[0][0] <= now - idle_minutes * 60  # the history reaches back a whole window
    if covered and window and all(u < idle_below for u in window):
        return f"stop: GPU under {idle_below}% for {idle_minutes} minutes"
    return None


def utilization():
    """the mean utilization over the pod's GPUs, or None when nvidia-smi does not answer"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        vals = [float(x) for x in out.split()]
        return sum(vals) / len(vals) if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_pod(dry_run):
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        print("watchdog: RUNPOD_POD_ID is not set (not on a RunPod pod): nothing to stop", flush=True)
        return False
    if dry_run:
        print(f"watchdog: would stop pod {pod}", flush=True)
        return True
    r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"watchdog: runpodctl stopped pod {pod}", flush=True)
        return True
    print(f"watchdog: runpodctl failed ({(r.stderr or r.stdout).strip()[:200]}); trying REST v2", flush=True)
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("watchdog: RUNPOD_API_KEY is not set: the pod could not be stopped", flush=True)
        return False
    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod}/action",
        data=json.dumps({"action": "stop"}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"watchdog: REST v2 stop answered {resp.status}", flush=True)
            return 200 <= resp.status < 300
    except OSError as e:
        print(f"watchdog: REST v2 stop failed: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, required=True, help="the chain to watch")
    ap.add_argument("--idle-minutes", type=float, default=20.0)
    ap.add_argument("--idle-below", type=float, default=10.0, help="a sample under this utilization (%%) counts as idle")
    ap.add_argument("--arm-above", type=float, default=50.0, help="idle stops count only after one sample above this (%%)")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between samples")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    samples, armed = [], False
    print(f"watchdog: watching pid {args.pid}; stop after {args.idle_minutes} min under {args.idle_below}% once armed above {args.arm_above}%", flush=True)
    while True:
        now = time.time()
        u = utilization()
        if u is not None:
            armed = armed or u > args.arm_above
            # the history starts when the GPU first works, and two windows of it are enough
            samples = [s for s in samples + [(now, u)] if s[0] >= now - 2 * args.idle_minutes * 60] if armed else []
        why = decide(samples, now, chain_alive=alive(args.pid), armed=armed, idle_minutes=args.idle_minutes, idle_below=args.idle_below)
        if why:
            print(f"watchdog: {time.strftime('%Y-%m-%d %H:%M:%S')} {why}", flush=True)
            sys.exit(0 if stop_pod(args.dry_run) else 1)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
