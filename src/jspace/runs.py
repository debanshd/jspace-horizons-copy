"""Run folders: one per experiment and model, with the log, the provenance, and the experiment's own outputs.

runs/<run name>/<model>/
    run.log        everything the experiment printed
    config.json    arguments, pinned revisions, package versions, GPU, code version
    ...            the experiment's CSV and JSON outputs, and stimuli.png (the first scenes as the model saw them)
"""

import argparse
import datetime
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_env(path=config.ROOT / ".env"):
    """KEY=value lines of .env into the environment (variables already set win): GEMINI_API_KEY, HF_TOKEN"""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("'\""))


def parser(description, run_name):
    """The arguments every experiment takes; experiments add their own. Also reads .env."""
    load_env()
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", choices=sorted(config.MODELS))
    ap.add_argument("--run", default=run_name, help=f"run name: outputs go to runs/<run>/<model>/ (default {run_name})")
    ap.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    return ap


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _json_default(o):
    """numpy scalars and arrays as JSON numbers and lists, so results read back with their types"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _cpu():
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(config.ROOT), "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+uncommitted" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class Run:
    """The output folder of one experiment on one model. Creating it starts the log."""

    def __init__(self, args, experiment):
        self.args, self.experiment = args, experiment
        self.dir = Path(args.runs_dir) / args.run / args.model.split("/")[-1]
        self.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.dir / "run.log", "a")
        sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
        print(f"===== {experiment} · {args.model} · {self.dir} · {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    def path(self, name):
        return self.dir / name

    def save_json(self, obj, name):
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        print("saved", self.dir / name)

    def save_csv(self, df: pd.DataFrame, name):
        df.to_csv(self.dir / name, index=False)
        print("saved", self.dir / name)

    def write_config(self, session):
        """Provenance: what ran, on what, at which revisions."""
        import torch
        import transformers

        try:
            from importlib.metadata import version

            fla = version("flash-linear-attention")
        except Exception:
            fla = None
        self.save_json(
            dict(
                experiment=self.experiment,
                argv=sys.argv,
                args={k: str(v) for k, v in vars(self.args).items()},
                model=session.name,
                model_revision=session.spec.revision,
                lens_repo=config.LENS_REPO,
                lens_revision=config.LENS_REVISION,
                lens_file=session.spec.lens_file,
                jpp_file=session.spec.jpp_file,
                read_layers=session.layers,
                batch=session.batch,
                batch_tokens=session.batch_tokens,
                batch_check=session.batch_check,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cpu=_cpu(),
                cpu_threads=torch.get_num_threads(),
                python=platform.python_version(),
                torch=torch.__version__,
                transformers=transformers.__version__,
                flash_linear_attention=fla,
                code=_git_commit(),
                date=datetime.datetime.now().isoformat(timespec="seconds"),
                env={k: v for k, v in os.environ.items() if k in ("CUDA_VISIBLE_DEVICES", "HF_HOME")},
            ),
            "config.json",
        )
