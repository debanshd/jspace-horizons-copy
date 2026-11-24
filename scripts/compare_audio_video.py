#!/usr/bin/env python
"""Each model's audio results beside its video results, input by input: the counterpart pairs of audio_sweep_static and
video_sweep2 (spoken / moving video, frozen word / frozen video, typed word / still picture). It reads two summary.json files
per model and computes nothing new: the verdicts, cuts and predictions are the runs' own.

    uv run scripts/compare_audio_video.py --audio runs/audio_sweep_static --video runs/video_sweep2

A model missing from either run is listed as missing; the table goes to stdout and, with --out, to a JSON file.
"""

import argparse
import json
from pathlib import Path

PAIRS = [("q_audio", "moving"), ("q_frozen", "frozen"), ("q_typed", "still")]


def cell(patch, inp):
    """an input's verdict, its L_say / L_use, and its predictions, from a summary's patch block"""
    if inp not in patch.get("verdict", {}):
        return dict(verdict="not run")
    cut = patch["cut"][inp]
    return dict(
        verdict=patch["verdict"][inp], L_say=cut.get("L_say") if isinstance(cut, dict) else None, L_use=cut.get("L_use") if isinstance(cut, dict) else None, predictions=patch["predictions"].get(inp)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True, help="an audio_sweep_static run folder (one subfolder per model)")
    ap.add_argument("--video", type=Path, required=True, help="a video_sweep2 run folder")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    models = sorted({p.parent.name for p in args.audio.glob("*/summary.json")} | {p.parent.name for p in args.video.glob("*/summary.json")})
    out = {}
    for m in models:
        a, v = args.audio / m / "summary.json", args.video / m / "summary.json"
        if not a.exists() or not v.exists():
            out[m] = f"missing {'audio' if not a.exists() else 'video'} run"
            print(f"{m}: {out[m]}")
            continue
        pa, pv = json.loads(a.read_text()).get("patch", {}), json.loads(v.read_text()).get("patch", {})
        out[m] = {f"{ai} / {vi}": dict(audio=cell(pa, ai), video=cell(pv, vi)) for ai, vi in PAIRS}
        print(m)
        for k, e in out[m].items():
            fa, fv = e["audio"], e["video"]
            print(f"  {k:20s} audio {fa['verdict']:28s} {fa.get('L_say')}/{fa.get('L_use')}   video {fv['verdict']:28s} {fv.get('L_say')}/{fv.get('L_use')}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
