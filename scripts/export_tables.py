#!/usr/bin/env python
"""Flat CSV tables of every result, for anyone drawing their own figures: one long-format table of every summary.json under
results/ (run, model, experiment, key path, value, lo, hi), plus tidy tables for the knockouts (costs per layer, stations), the
single-layer patching battery (depths and curves) and the access profiles. Written to results/tables/.
    .venv/bin/python scripts/export_tables.py
"""

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables"


def is_triple(v):
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def flatten(obj, prefix=""):
    """(key path, value, lo, hi) rows; a [point, lo, hi] list becomes one row with its interval"""
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows += flatten(v, f"{prefix}/{k}" if prefix else str(k))
    elif is_triple(obj):
        rows.append((prefix, obj[0], obj[1], obj[2]))
    elif isinstance(obj, list):
        if all(x is None or isinstance(x, (int, float, str, bool)) for x in obj):
            rows.append((prefix, json.dumps(obj), "", ""))
        else:
            for i, v in enumerate(obj):
                rows += flatten(v, f"{prefix}[{i}]")
    else:
        rows.append((prefix, obj, "", ""))
    return rows


def summaries():
    """every (run, model, experiment, summary dict) under results/"""
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / "oct21" / "*" / "*" / "summary.json"))):
        p = Path(f)
        cfg = p.parent / "config.json"
        exp = json.load(open(cfg)).get("experiment", p.parent.parent.name) if cfg.exists() else p.parent.parent.name
        out.append((p.parent.parent.name, p.parent.name, exp, json.load(open(f))))
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / "*" / "summary.json"))):
        p = Path(f)
        if p.parent.parent.name in ("oct21", "samples", "examples", "tables"):
            continue
        out.append((p.parent.parent.name, p.parent.name, p.parent.parent.name, json.load(open(f))))
    return out


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote results/tables/{name} ({len(rows)} rows)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = summaries()
    rows = []
    for run, model, exp, s in S:
        for key, v, lo, hi in flatten(s):
            rows.append((run, model, exp, key, v, lo, hi))
    write("all_summaries.csv", ["run", "model", "experiment", "key", "value", "lo", "hi"], rows)
    # the knockouts: cost per layer and kind, and the stations
    costs, stations = [], []
    for run, model, exp, s in S:
        if exp != "attention_knockout":
            continue
        n_layers = s.get("n_layers") or (max(s["attention_layers"]) + 1)
        for t, q in s["questions"].items():
            if not isinstance(q, dict) or "single" not in q:
                continue
            for kind in ("single", "from", "upto"):
                for L, v in q.get(kind, {}).items():
                    c = v.get("cost", [None, None, None])
                    cr = v.get("correct", [None, None, None])
                    costs.append((run, model, n_layers, t, q.get("needs"), q.get("answer_type"), kind, int(L), c[0], c[1], c[2], cr[0]))
            ab = q.get("all_blocked_ld", [None, None, None])
            stations.append(
                (
                    run,
                    model,
                    n_layers,
                    t,
                    q.get("needs"),
                    q.get("answer_type"),
                    q.get("clean_correct"),
                    (q.get("clean_ld") or [None])[0],
                    json.dumps(q.get("readers")),
                    q.get("last_reader"),
                    ab[0],
                    q.get("complete"),
                )
            )
    write("knockout_costs.csv", ["run", "model", "n_layers", "question", "needs", "answer_type", "kind", "layer", "cost_nats", "lo", "hi", "correct"], costs)
    write("knockout_stations.csv", ["run", "model", "n_layers", "question", "needs", "answer_type", "clean_correct", "clean_ld", "readers", "last_reader", "all_blocked_ld", "complete"], stations)
    # the single-layer patching battery: depths and the curves
    depths, curves = [], []
    for run, model, exp, s in S:
        if exp != "read_battery":
            continue
        for t, q in s["questions"].items():
            if not isinstance(q, dict) or "depth_ld" not in q:
                continue
            depths.append(
                (
                    run,
                    model,
                    t,
                    q.get("needs"),
                    q.get("answer_type"),
                    q.get("gate_pass"),
                    q.get("clean_keeps_Z"),
                    (q.get("clean_ld") or [None])[0],
                    q.get("depth_flip"),
                    q.get("depth_ld"),
                    json.dumps(q.get("transition")),
                )
            )
            ld_by, fl_by, sweep = q.get("ld_by_layer") or {}, q.get("flip_by_layer") or {}, s.get("sweep") or []
            if isinstance(ld_by, list):  # aligned with the summary's sweep of patch layers
                ld_by = dict(zip(sweep, ld_by))
            if isinstance(fl_by, list):
                fl_by = dict(zip(sweep, fl_by))
            for L, v in ld_by.items():
                fl = fl_by.get(L)
                v3 = v if is_triple(v) else [v, None, None]
                f3 = fl if is_triple(fl) else [fl, None, None]
                curves.append((run, model, t, int(L), v3[0], v3[1], v3[2], f3[0], f3[1], f3[2]))
    write("battery_depths.csv", ["run", "model", "question", "needs", "answer_type", "gate_pass", "clean_keeps_Z", "clean_ld", "depth_flip", "depth_ld", "transition"], depths)
    write("battery_curves.csv", ["run", "model", "question", "patch_layer", "ld", "lo", "hi", "flip_share", "flip_lo", "flip_hi"], curves)
    acc = ROOT / "results" / "figures" / "figAccess.csv"
    if acc.exists():
        (OUT / "access_profiles.csv").write_text(acc.read_text())
        print("wrote results/tables/access_profiles.csv")
    (OUT / "README.md").write_text(
        "# Tables\n\n"
        "Flat CSVs for drawing your own figures, written by scripts/export_tables.py from the summaries under results/.\n\n"
        "- all_summaries.csv: every value of every summary.json (run, model, experiment, key path, value, and the 95% scene-bootstrap interval where the "
        "summary carries one). Grep the key path for what you need, for example `arms/all heads of 7+11/names_X` or `questions/animal/from/11/cost`.\n"
        "- knockout_costs.csv: attention knockouts, the cost in nats of blocking one layer (single), every layer from L on (from) or up to L (upto), per model and question, with intervals.\n"
        "- knockout_stations.csv: per model and question, the readers (layers whose block costs 1 nat or more), "
        "the last reader, the clean log-probability difference and what blocking every layer leaves.\n"
        "- battery_depths.csv and battery_curves.csv: single-layer patching with log-probabilities (read_battery), "
        "the read depth per question by flips and by log-probability, and the full curve over patch layers.\n"
        "- access_profiles.csv: per model and question class, the depth by which half the read is done, the window width in depth and in layers, and the largest share any single layer carries.\n\n"
        "Per-scene answers for every run are the CSVs under results/oct21/<run>/<model>/; the 14 October per-scene outputs are under results/<experiment>/<model>/.\n"
    )
    print("wrote results/tables/README.md")


if __name__ == "__main__":
    main()
