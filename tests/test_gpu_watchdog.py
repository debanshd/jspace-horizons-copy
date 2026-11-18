"""scripts/gpu_watchdog.py's rule: when it stops the pod and when it must not (CPU only, no GPU)"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_watchdog import decide  # noqa: E402

KW = dict(idle_minutes=20, idle_below=10)


def series(start, minutes, util):
    return [(start + 30 * i, util) for i in range(int(minutes * 2))]


def test_chain_exit_stops_even_before_the_gpu_worked():
    assert decide([], 0.0, chain_alive=False, armed=False, **KW) == "stop: the chain has exited"


def test_the_cpu_only_start_never_stops():
    # downloads and clip decoding: an hour at 0% before the GPU first works
    s = series(0, 60, 0.0)
    assert decide(s, s[-1][0], chain_alive=True, armed=False, **KW) is None


def test_idle_for_the_whole_window_stops():
    s = series(0, 30, 95.0) + series(30 * 60, 21, 2.0)
    assert decide(s, s[-1][0], chain_alive=True, armed=True, **KW).startswith("stop: GPU under 10%")


def test_a_gap_between_runs_does_not_stop():
    # a model loading and its prompts encoded: 8 minutes idle, then busy again
    s = series(0, 30, 95.0) + series(30 * 60, 8, 0.0) + series(38 * 60, 1, 97.0)
    assert decide(s, s[-1][0], chain_alive=True, armed=True, **KW) is None
    s = series(0, 30, 95.0) + series(30 * 60, 15, 0.0)  # still inside the window
    assert decide(s, s[-1][0], chain_alive=True, armed=True, **KW) is None


def test_one_busy_sample_in_the_window_keeps_the_pod():
    s = series(0, 30, 95.0) + series(30 * 60, 10, 1.0) + [(40 * 60, 60.0)] + series(40 * 60 + 30, 12, 1.0)
    assert decide(s, s[-1][0], chain_alive=True, armed=True, **KW) is None


def test_a_history_shorter_than_the_window_does_not_stop():
    # armed a minute ago and idle since: not 20 minutes of evidence yet
    s = [(0.0, 90.0), (30.0, 1.0), (60.0, 1.0)]
    assert decide(s, 60.0, chain_alive=True, armed=True, **KW) is None


def test_dry_run_off_the_pod_stops_nothing(tmp_path):
    # a chain that exits at once; off a RunPod pod (no RUNPOD_POD_ID) the watchdog says so and stops nothing
    chain = subprocess.Popen([sys.executable, "-c", "pass"])
    chain.wait()
    env = {"PATH": "/usr/bin:/bin"}
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/gpu_watchdog.py"), "--pid", str(chain.pid), "--every", "1", "--dry-run"], capture_output=True, text=True, env=env, timeout=60)
    assert "the chain has exited" in r.stdout and "nothing to stop" in r.stdout, r.stdout + r.stderr
    assert time.time() - t0 < 30
