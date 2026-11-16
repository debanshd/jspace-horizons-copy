#!/usr/bin/env python
"""Compare two media_steer run folders answer by answer (plan.md step 6: the smoke check run twice must give the same outputs).

    uv run scripts/compare_media.py runs/smoke_media_steer runs/smoke_media_steer_repeat

For every model folder in the first run: media_steer.csv row by row with check.py's rule (the same rows in the same order,
identical answers, numbers within --tol). Exit status 1 when anything differs or a folder is missing.
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
    args = ap.parse_args()
    folders = sorted(p.parent.name for p in args.a.glob("*/media_steer.csv"))
    if not folders:
        raise SystemExit(f"no media_steer.csv under {args.a}")
    bad = 0
    for m in folders:
        other = args.b / m / "media_steer.csv"
        if not other.exists():
            print(f"missing  {other}")
            bad += 1
            continue
        ok, note = compare_csv(args.a / m / "media_steer.csv", other, args.tol)
        print(f"{'same' if ok else 'DIFFERS'}  {m}: {note}")
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
