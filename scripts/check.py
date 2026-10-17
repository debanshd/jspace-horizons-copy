#!/usr/bin/env python
"""Compare a reproduction with the original runs: every file the example figures are made from, row by row.

    uv run scripts/check.py                     # runs/ against results/
    uv run scripts/check.py --runs some/dir     # any folder with the same layout

Per CSV: the same rows in the same order, identical answers and judge verdicts, and numbers within --tol. Also the crop gate
(crops.json), the stimulus sheets (stimuli.png, pixel for pixel) and examples/outputs.csv. An output that has not been produced
yet is listed as missing, not as a difference. Exit status 1 when anything differs.
On the GPU type and software of the original runs the outputs are expected to be identical; on other GPUs, floating-point
differences can change a few answers (see README).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from jspace.config import RESULTS_DIR, RUNS_DIR

CSVS = [
    "printed_word/Qwen3.5-4B/printed_word.csv",
    "printed_word/Qwen3.5-4B/scenes.csv",
    "scam_notes/Qwen3.5-4B/scam_notes.csv",
    "scam_notes/Qwen3.5-4B/items.csv",
    "name_steering/Qwen3.5-4B/name_steering.csv",
    "name_steering/Qwen3.5-4B/logp.csv",
    "subspace_patching/Qwen3.5-4B/subspace_patching.csv",
    "depth_sweep/Qwen3.5-4B/depth_sweep.csv",
    "depth_sweep_spectrum/Qwen3.5-4B/depth_sweep.csv",
    "back_patching/Qwen3.5-4B/back_patching.csv",
    "gqa_ablation/Qwen3.5-4B/gqa_ablation.csv",
    "subspace_patching/gemma-4-31B-it/subspace_patching.csv",
    "depth_sweep/gemma-4-31B-it/depth_sweep.csv",
    "examples/outputs.csv",
]
SHEETS = [p for p in sorted(RESULTS_DIR.glob("*/*/stimuli.png"))]


def as_text(s):
    """values as text, a missing value (NaN, None) as one marker, so that two missing values are equal"""
    return s.map(lambda v: "<missing>" if pd.isna(v) else str(v)).astype(object)


def compare_csv(a_path, b_path, tol):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return False, f"{len(a)} rows, expected {len(b)}"
    bad, notes = 0, []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            d = np.abs(x.to_numpy(float) - y.to_numpy(float))
            both_nan = np.isnan(x.to_numpy(float)) & np.isnan(y.to_numpy(float))
            d = np.where(both_nan, 0, d)
            n = int((~(d <= tol)).sum())
            if n:
                bad += n
                notes.append(f"{c}: {n} values differ by more than {tol} (max {np.nanmax(d):.2g})")
        else:
            differ = (as_text(x) != as_text(y)).to_numpy()
            n = int(differ.sum())
            if n:
                bad += n
                first = int(np.flatnonzero(differ)[0])
                notes.append(f"{c}: {n} of {len(a)} differ (first at row {first}: {x.iloc[first]!r}, expected {y.iloc[first]!r})")
    return bad == 0, "; ".join(notes) if notes else f"identical ({len(a)} rows)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--reference", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerance for numeric columns")
    args = ap.parse_args()
    ok_all, missing = True, []
    for rel in CSVS:
        mine, ref = args.runs / rel, args.reference / rel
        if not mine.exists():
            missing.append(rel)
            continue
        ok, note = compare_csv(mine, ref, args.tol)
        ok_all &= ok
        print(f"{'same' if ok else 'DIFF'}  {rel}: {note}")
    for ref in sorted(args.reference.glob("*/*/crops.json")):
        rel = ref.relative_to(args.reference)
        mine = args.runs / rel
        if mine.exists():
            ok = json.load(open(mine))["crop_gate"] == json.load(open(ref))["crop_gate"]
            ok_all &= ok
            print(f"{'same' if ok else 'DIFF'}  {rel}: crop gate {'identical' if ok else 'differs'}")
    for ref in SHEETS:
        rel = ref.relative_to(RESULTS_DIR)
        mine = args.runs / rel
        if mine.exists():
            ok = np.array_equal(np.asarray(Image.open(mine).convert("RGB")), np.asarray(Image.open(args.reference / rel).convert("RGB")))
            ok_all &= ok
            print(f"{'same' if ok else 'DIFF'}  {rel}: {'pixel-identical' if ok else 'pixels differ'}")
    for rel in missing:
        print(f"----  {rel}: not produced yet")
    print("everything produced so far matches the original runs" if ok_all else "some outputs differ from the original runs")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
