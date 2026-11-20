#!/usr/bin/env python
"""Rescore saved video_sweep2 answers with strict.py's whole-answer scoring, so the video and the audio results are read under
one scoring policy. Nothing is run on a model; the saved answers are read and the original outputs are left as they are.

    uv run scripts/rescore_strict.py --source runs/video_sweep2 --run rescore_strict_video_sweep2

Per model folder of --source (video_sweep2.csv, clips.json, summary.json), into runs/<run>/<model>/:
- The answer scored is the first answer (first_answer where video_sweep2 asked its hedge follow-up, else generation): the
  follow-up is not part of this policy.
- say is scored at strict.py's synonym level (the whole answer is the label, its plural, or a kind or spelling of it): a picture
  is not a spoken word, so there is no exact word to recognize. letter, set, animal and ride: the whole answer is the one letter
  or digit. The saved answers do not record whether they hit the 4-token limit, so none is marked as cut: an answer cut off
  exactly after a whole label would still count, which this rescoring cannot rule out.
- video_sweep's patch summary (gates, controls, horizons, cut, verdicts, P1 to P5) recomputed on the rescored rows, beside the
  saved summary's, and how many rows changed score. The steering summaries are not recomputed.
- A sensitivity level, reported apart: say also counts a phrase of at most three words whose last word is the label, with no
  list and no negation (strict.phrase: "sports car", "wall clock"). clip_naming2 credited such head nouns; whole-answer
  scoring at the synonym level does not, and the two can give different verdicts.
Outputs: summary.json (saved against rescored, per input), changed.csv (every patching row whose score changed), run.log.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import video_sweep as vs  # noqa: E402  the patch summary

from jspace import strict  # noqa: E402
from jspace.config import RUNS_DIR  # noqa: E402
from jspace.runs import Run  # noqa: E402

PATCH = ["pclean", "self", "full", "transplant", "inject", "random"]


def judge(task, text, word, membership, level="synonym"):
    """strict.py's policy for a picture or a video: say at the synonym level (or the phrase level), the rest whole-answer"""
    if task == "say":
        return strict.say(text, word)[1] or (level == "phrase" and strict.phrase(text, word))
    return strict.matches(task, text, word, membership)


def rescore(R, sets, level="synonym"):
    """the patching rows of R, scored on the first answer by judge; the saved scores kept as old_*"""
    R = R[R["family"].isin(PATCH)].copy()
    first = R["first_answer"].astype(object).fillna("").astype(str) if "first_answer" in R else pd.Series("", index=R.index)
    R["scored"] = first.where(first != "", R["generation"].fillna("").astype(str))
    for col, w in (("correct", "X"), ("follows_Y", "Y"), ("follows_Z", "Z")):
        R[f"old_{col}"] = R[col].astype(bool)
        R[col] = [judge(t, a, x, sets[s], level) for t, a, x, s in zip(R["task"], R["scored"], R[w], R["scene"])]
    R["valid"] = [(strict.say(a, x)[1] or strict.valid(t, a)) if t == "say" else strict.valid(t, a) for t, a, x in zip(R["task"], R["scored"], R["X"])]
    return R


def n_layers_of(R):
    """the full-donor rows patch layers -1 to n_layers - 1"""
    last = R[R["family"] == "full"]["layers"].iloc[0]
    return int(str(last).split("..")[1]) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="a video_sweep2 run folder (one subfolder per model)")
    ap.add_argument("--run", default="rescore_strict_video_sweep2")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = ap.parse_args()
    folders = sorted(p.parent for p in args.source.glob("*/video_sweep2.csv"))
    if not folders:
        raise SystemExit(f"no video_sweep2.csv under {args.source}")
    for f in folders:
        for need in ("clips.json", "summary.json"):
            if not (f / need).exists():
                raise SystemExit(f"{f / need} not found")
        saved = json.loads((f / "summary.json").read_text())
        if saved.get("smoke"):
            print(f"{f.name}: a smoke check, skipped")
            continue
        run = Run(argparse.Namespace(model=f.name, run=args.run, runs_dir=args.runs_dir), "rescore_strict")
        sets = {r["scene"]: r["set"] for r in json.loads((f / "clips.json").read_text())["scenes"]}
        raw = pd.read_csv(f / "video_sweep2.csv", low_memory=False)
        R = rescore(raw, sets)
        inputs = saved["setup"]["inputs"]
        args_ps = (inputs, n_layers_of(R), saved["setup"]["family"], saved["setup"]["kv_limit"])
        new = vs.patch_summary(R, *args_ps)
        ph = vs.patch_summary(rescore(raw, sets, "phrase"), *args_ps)
        old = saved["patch"]
        changed = R[(R["correct"] != R["old_correct"]) | (R["follows_Y"] != R["old_follows_Y"]) | (R["follows_Z"] != R["old_follows_Z"])]
        out = dict(
            source=str(f), rows=len(R), changed=len(changed), changed_by_task=changed.groupby("task").size().to_dict(), invalid_by_task=(~R["valid"]).groupby(R["task"]).sum().astype(int).to_dict()
        )
        for inp in inputs:
            same = lambda a, b: a == json.loads(json.dumps(b, default=float))  # noqa: E731
            out[inp] = {
                k: dict(saved=old[k][inp], rescored=new[k][inp], same=same(old[k][inp], new[k][inp]), phrase=ph[k][inp], phrase_same=same(old[k][inp], ph[k][inp]))
                for k in ("gate", "controls", "horizon", "cut", "verdict", "predictions")
            }
        run.save_json(out, "summary.json")
        run.save_csv(changed[["input", "task", "scene", "X", "Y", "Z", "family", "arm", "scored", "old_correct", "correct", "old_follows_Y", "follows_Y", "old_follows_Z", "follows_Z"]], "changed.csv")
        print(f"{f.name}: {len(changed)} of {len(R)} patching rows change score ({out['changed_by_task']})")
        for inp in inputs:
            say = [s["gate"][inp]["say"]["clean"] for s in (old, new, ph)]
            print(f"  {inp}: verdict saved {old['verdict'][inp]!r}, rescored {new['verdict'][inp]!r}, phrase level {ph['verdict'][inp]!r}; clean say {say[0]:.2f} -> {say[1]:.2f} ({say[2]:.2f})")


if __name__ == "__main__":
    main()
