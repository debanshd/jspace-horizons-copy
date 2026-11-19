#!/usr/bin/env python
"""Compare two media_steer run folders answer by answer (plan.md step 6: the smoke check run twice must give the same outputs).

    uv run scripts/compare_media.py runs/smoke_media_steer runs/smoke_media_steer_repeat

For every model folder in the first run (or only --model's): media_steer.csv row by row with check.py's rule (the same rows in
the same order, identical answers, numbers within --tol). Exit status 1 when anything differs or a folder is missing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import compare_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerance for numeric columns")
    ap.add_argument("--file", default="media_steer.csv", help="the CSV each model folder holds (video_sweep.csv for video_sweep)")
    # one model only: pods sharing a volume run their models at once, and another model's repeat may not have finished yet
    ap.add_argument("--model", help="compare this model folder only (e.g. Qwen3.5-4B); default every model folder of the first run")
    args = ap.parse_args()
    folders = sorted(p.parent.name for p in args.a.glob(f"*/{args.file}"))
    if args.model:
        folders = [m for m in folders if m == args.model]
    if not folders:
        raise SystemExit(f"no {args.file} under {args.a}" + (f" for {args.model}" if args.model else ""))
    bad = 0
    for m in folders:
        other = args.b / m / args.file
        if not other.exists():
            print(f"missing  {other}")
            bad += 1
            continue
        ok, note = compare_csv(args.a / m / args.file, other, args.tol)
        print(f"{'same' if ok else 'DIFFERS'}  {m}: {note}")
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
