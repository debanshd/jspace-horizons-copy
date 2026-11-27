"""The verdicts of jpp_wsp/RULE.md, from the pod's readout ranks.

Recomputes the harness's item-weighted recall@k (per item, the fraction of its intermediates whose best rank over the seven layers is
<= k; per task, the mean over items; macro, the mean over the five tasks) and checks it against the harness's own table. Then the
paired bootstrap of the rule: items resampled within each task, 2000 resamples, numpy default_rng(0).

    python jpp_wsp/verdicts.py --out jpp_wsp/out/olmo-3-7b-50pct --stamp100 <olmo7b_jpp_n64_stamp.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["multihop", "multilingual", "typo", "association", "poetry"]
TARGET, PRIMARY, CONTROL = "olmo7b50pct_jpp_n64", "wspV1_jpp_n64", "wspV0_jpp_n64"
MARGIN = 2.0  # points of recall@10, fixed in the rule
N_BOOT, SEED = 2000, 0


def item_hits(ranks: pd.DataFrame, k: int) -> pd.DataFrame:
    """one row per (eval, item), one column per lens: the item's recall@k"""
    best = ranks.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = best["rank"] <= k
    per_item = best.groupby(["eval", "lens", "item"])["hit"].mean()
    return per_item.unstack("lens")


def macro(hits: pd.DataFrame) -> pd.Series:
    return 100 * hits.groupby(level="eval").mean().reindex(TASKS).mean()


def bootstrap_diff(hits: pd.DataFrame, a: str, b: str) -> tuple[float, float, float]:
    """macro(a) - macro(b), and its 95% interval over item resamples within each task"""
    rng = np.random.default_rng(SEED)
    per_task = [hits.loc[t, [a, b]].to_numpy(dtype=float) for t in TASKS]
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means = [x[rng.integers(0, len(x), len(x))].mean(axis=0) for x in per_task]
        m = np.mean(means, axis=0)
        diffs[i] = 100 * (m[0] - m[1])
    point = float(macro(hits)[a] - macro(hits)[b])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="the pod's output folder of the 50% model")
    ap.add_argument("--stamp100", type=Path, required=True, help="the 100% J++ fit's stamp, for the items its weights saw")
    ap.add_argument("--primary", default=PRIMARY, help="the primary lens: wspV1_jpp_n64 (RULE.md) or wspP1_jpp_n64 (RULE2.md)")
    ap.add_argument("--name", default="verdicts.json", help="the output file, beside --out")
    ap.add_argument("--also", nargs="*", default=[], help="lenses to compare with the primary, reported without a verdict")
    args = ap.parse_args()
    primary = args.primary

    report = {}
    for mode in ("filtered", "full"):
        ranks = pd.read_csv(args.out / f"readout_{mode}.csv")
        harness = pd.read_csv(args.out / f"readout_{mode}_pass_at_k.csv")
        harness = harness[(harness["weighting"] == "item") & (harness["eval"] == "macro")]
        for k in (1, 10):
            hits = item_hits(ranks, k)
            ours = macro(hits)
            theirs = 100 * harness[harness["k"] == k].set_index("lens")["pass_at_k"]
            gap = (ours - theirs.reindex(ours.index)).abs().max()
            if gap > 1e-9:
                raise SystemExit(f"{mode} recall@{k}: this recomputation differs from the harness's by {gap}")
            report[f"{mode}_macro_r@{k}"] = ours.round(2).to_dict()

    hits = item_hits(pd.read_csv(args.out / "readout_filtered.csv"), 10)
    report["n_items"] = {t: int(len(hits.loc[t])) for t in TASKS}
    r10 = macro(hits)
    d_help = bootstrap_diff(hits, primary, CONTROL)
    d_match = bootstrap_diff(hits, primary, TARGET)
    d_transfer = bootstrap_diff(hits, CONTROL, TARGET)
    report["primary"] = primary
    report["diff_primary_minus_V0"] = d_help
    report["diff_primary_minus_target"] = d_match
    for other in args.also:
        report[f"diff_primary_minus_{other}"] = bootstrap_diff(hits, primary, other)
    report["diff_V0_minus_target"] = d_transfer

    if d_help[0] >= MARGIN and d_help[1] > 0:
        helps = "the edit helps"
    elif d_help[2] < 0:
        helps = "the edit harms"
    else:
        helps = "the edit does not help"
    report["verdict_1"] = helps
    report["verdict_2"] = "matches the fitted lens" if d_match[0] >= -MARGIN else "does not match the fitted lens"
    report["verdict_3"] = "the 100% lens already transfers" if d_transfer[0] >= -MARGIN else "the 100% lens does not transfer"

    # secondary: the held-out items the 100% lens's expert weights did not see either
    seen = set(json.loads(args.stamp100.read_text())["fitting_item_names"])
    unseen = hits[~hits.index.get_level_values("item").isin(seen)]
    report["unseen_by_100pct_n_items"] = {t: int(len(unseen.loc[t])) if t in unseen.index.get_level_values("eval") else 0 for t in TASKS}
    report["unseen_by_100pct_filtered_macro_r@10"] = macro(unseen).round(2).to_dict()

    report["filtered_macro_r@10_primary_table"] = r10.round(2).to_dict()
    (args.out.parent / args.name).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
