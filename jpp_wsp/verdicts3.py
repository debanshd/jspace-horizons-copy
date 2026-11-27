"""The reading of RULE3.md: jeeva2812's J-lens against the lenses scored before on the same held-out split of the 100% model.

The new run (out3/) scored jeeva's lens and the logit lens; the earlier run (the 100% J++ fit's out_fits/olmo-3-7b/) scored the logit
lens, Neuronpedia's J-lens and the J++ lens. The logit lens, in both, checks that the two runs rank alike. Paired bootstrap as in
verdicts.py, over the items both runs scored.

    python jpp_wsp/verdicts3.py --new jpp_wsp/out3/olmo-3-7b --old <out_fits/olmo-3-7b>
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from verdicts import TASKS, bootstrap_diff, item_hits, macro

JEEVA, NP, JPP = "jeeva_Jall_main", "np_olmo-3-7b_jlens", "olmo7b_jpp_n64"
MARGIN = 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    for mode in ("filtered", "full"):
        new = pd.read_csv(args.new / f"readout_{mode}.csv")
        old = pd.read_csv(args.old / f"readout_{mode}.csv")
        old["lens"] = old["lens"].replace({"logit": "logit_earlier_run"})
        ranks = pd.concat([new, old], ignore_index=True)
        for k in (1, 10):
            hits = item_hits(ranks, k)
            report[f"{mode}_macro_r@{k}"] = macro(hits).round(2).to_dict()
            if mode == "filtered" and k == 10:
                both = hits.dropna()
                report["n_items_both_runs"] = {t: int(len(both.loc[t])) for t in TASKS}
                report["n_items_new"] = {t: int(hits[JEEVA].loc[t].notna().sum()) for t in TASKS}
                # the two runs rank alike when the logit lens's per-item hits agree
                agree = (both["logit"] == both["logit_earlier_run"]).mean()
                report["logit_item_agreement"] = float(agree)
                for other in (NP, JPP, "logit"):
                    report[f"diff_jeeva_minus_{other}"] = bootstrap_diff(both, JEEVA, other)
    d = report["diff_jeeva_minus_" + NP]
    report["reading"] = "as good as Neuronpedia's J-lens" if d[0] >= -MARGIN else "below Neuronpedia's J-lens"
    swap = pd.read_csv(args.new / "probe_swap_best_scale.csv")
    report["probe_swap"] = swap.to_dict(orient="records")
    (args.new.parent / "verdicts3.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
