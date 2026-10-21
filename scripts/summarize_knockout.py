"""Rebuild summary.json of an attention_knockout run from its CSV (the same rules as the experiment; used when a run died after saving its CSV).

uv run scripts/summarize_knockout.py runs/run7k/Qwen3.5-4B
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from read_battery import QUESTIONS  # noqa: E402

from jspace.analysis import boot_scene  # noqa: E402


def main(folder):
    folder = Path(folder)
    R = pd.read_csv(folder / "attention_knockout.csv")
    AL = sorted(int(L) for L in R[R.kind == "single"].layer.dropna().unique())
    out = dict(n=int(R.scene.nunique()), n_layers=None, attention_layers=AL, questions={}, rebuilt_from_csv=True)
    for task in R.task.unique():
        d = R[(R.task == task) & R.distinct]
        scenes = np.sort(d.scene.unique())
        clean = d[d.kind == "clean"].set_index("scene").reindex(scenes)
        e = dict(needs=QUESTIONS[task][1], answer_type=QUESTIONS[task][2], n=int(len(scenes)), clean_correct=float(clean.correct.mean()), clean_ld=boot_scene(clean.ld.values, scenes))
        for kind_ in ("single", "from", "upto"):
            e[kind_] = {}
            for L in AL:
                g = d[(d.kind == kind_) & (d.layer == L)].set_index("scene").reindex(scenes)
                if len(g.dropna(subset=["ld"])) == 0:
                    continue
                e[kind_][L] = dict(correct=boot_scene(g.correct.astype(float).values, scenes), ld=boot_scene(g.ld.values, scenes), cost=boot_scene((clean.ld - g.ld).values, scenes))
        have = [L for L in AL if L in e["single"] and L in e["from"]]
        e["readers"] = [L for L in have if e["single"][L]["cost"][0] >= 1 and e["single"][L]["cost"][1] > 0]
        e["all_blocked_correct"] = e["from"][AL[0]]["correct"][0] if AL[0] in e["from"] else None
        e["all_blocked_ld"] = e["from"][AL[0]]["ld"] if AL[0] in e["from"] else None
        e["last_reader"] = max([L for L in have if e["from"][L]["cost"][0] >= 1 and e["from"][L]["cost"][1] > 0], default=None)
        e["complete"] = len(have) == len(AL) and all(L in e["upto"] for L in AL)
        out["questions"][task] = e
    json.dump(out, open(folder / "summary.json", "w"), indent=2)
    for t, v in out["questions"].items():
        ab = f"{v['all_blocked_ld'][0]:+.1f}" if v["all_blocked_ld"] else "n/a"
        print(f"{t:9s} clean {v['clean_correct']:.2f} ld {v['clean_ld'][0]:+.1f} | readers {v['readers']} last {v['last_reader']} | all blocked {ab} | complete {v['complete']}")


if __name__ == "__main__":
    main(sys.argv[1])
