"""Re-score the saved "say" answers of the original depth sweeps (runs/run4/<model>/d1_depth.csv, D1) with the frozen LLM judge,
beside the first-word string score the headlines used, and report where they disagree. No GPU: the generations are on disk.

    uv run scripts/judge_d1.py --models Qwen3.5-4B,Qwen3.5-2B,Qwen3.5-0.8B,gemma-4-31B-it,gemma-4-E4B-it

Writes runs/judge_d1/<model>/d1_say_judged.csv (every say row with the judge's verdict for X and for Z) and summary.json
(agreement with the string score, the horizon of the photo donor by each score).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from jspace.judge import Judge
from jspace.runs import load_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("/workspace/nayan/runs/run4"))
    ap.add_argument("--out", type=Path, default=Path("runs/judge_d1"))
    ap.add_argument("--models", default="Qwen3.5-4B,Qwen3.5-2B,Qwen3.5-0.8B,gemma-4-31B-it,gemma-4-E4B-it")
    args = ap.parse_args()
    load_env()
    for model in args.models.split(","):
        src = args.runs / model / "d1_depth.csv"
        if not src.exists():
            print("missing", src)
            continue
        out = args.out / model
        out.mkdir(parents=True, exist_ok=True)
        D = pd.read_csv(src)
        D = D[D.task == "say"].copy()
        D["generation"] = D.generation.fillna("").astype(str)
        judge = Judge(out, reference_caches=[Path("results/judge_cache.json")])
        D["judge_X"] = judge.name_match(D.generation.tolist(), D.X.tolist())
        D["judge_Z"] = judge.name_match(D.generation.tolist(), D.Z.tolist())
        D.to_csv(out / "d1_say_judged.csv", index=False)
        agree_x = float((D.judge_X.astype("boolean") == D.follows_X.astype(bool)).mean())
        dis = D[D.judge_X.astype("boolean") != D.follows_X.astype(bool)]
        summ = dict(model=model, n_rows=int(len(D)), judge_calls=judge.n_calls, judge_failed=judge.n_failed, agreement_follows_X=agree_x, disagreements=int(len(dis)))
        # the photo donor's naming horizon under each score: the last window start with >= 0.5
        P = D[D.kind == "transplant"]
        hz = {}
        for col in ("follows_X", "judge_X"):
            curve = P.groupby("w0")[col].apply(lambda v: v.astype("boolean").fillna(False).mean())
            hz[col] = {int(w): float(v) for w, v in curve.items()}
            summ[f"horizon_{col}"] = max([w for w, v in curve.items() if v >= 0.5], default=None)
        summ["curves"] = hz
        summ["disagreement_examples"] = dis[["arm", "scene", "X", "Z", "generation", "follows_X", "judge_X"]].head(20).to_dict("records")
        json.dump(summ, open(out / "summary.json", "w"), indent=2)
        hs, hj = summ["horizon_follows_X"], summ["horizon_judge_X"]
        print(f"{model}: {len(D)} say rows, judge agrees with the string score on {agree_x:.3f}; horizon by string {hs}, by judge {hj}; calls {judge.n_calls}, failed {judge.n_failed}")


if __name__ == "__main__":
    main()
