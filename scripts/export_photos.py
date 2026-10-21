#!/usr/bin/env python
"""The natural photos the runs used, at the size the models received them (stimuli.fit_photo), with their questions and answers, so
figures about them can be drawn without the datasets: results/samples/gqa (the 48 testdev images of run9 and run9p), results/samples/
countbench (the 400 photos the knockout kept) and results/samples/scam (the 64 photos of the 14 October note test, where the data is present).
    .venv/bin/python scripts/export_photos.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    """a run's CSV, stored as .csv or, in results/, as .csv.gz"""
    p = Path(path)
    return pd.read_csv(p if p.exists() else p.with_name(p.name + ".gz"))


sys.path.insert(0, str(ROOT / "src"))
from jspace import stimuli  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "samples"


def gqa():
    d = read_csv(ROOT / "results" / "oct21" / "run9p" / "Qwen3.5-4B" / "gqa_knockout.csv")
    q = d[d.kind == "clean"][["imageId", "qkind", "structural", "sem", "question", "reference"]].drop_duplicates()
    tab = pd.read_parquet(DATA / "gqa" / "testdev_balanced_images" / "testdev-00000-of-00001.parquet")
    tab = tab[tab.id.isin(q.imageId.unique())].set_index("id")
    out = OUT / "gqa"
    out.mkdir(parents=True, exist_ok=True)
    for iid in q.imageId.unique():
        stimuli.fit_photo(tab.loc[iid, "image"]["bytes"]).save(out / f"{iid}.png")
    q.to_csv(out / "questions.csv", index=False)
    print(f"wrote {out}: {q.imageId.nunique()} photos, {len(q)} questions")


def countbench():
    d = read_csv(ROOT / "results" / "oct21" / "run10" / "Qwen3.5-4B" / "bench_knockout.csv")
    cp = read_csv(ROOT / "results" / "oct21" / "run10" / "Qwen3.5-4B" / "clean_pass.csv")
    items = sorted(d.item.unique())
    tab = pd.read_parquet(DATA / "countbench" / "data" / "test-00000-of-00001.parquet")
    out = OUT / "countbench"
    out.mkdir(parents=True, exist_ok=True)
    for it in items:
        stimuli.fit_photo(tab.iloc[int(it)]["image"]["bytes"]).save(out / f"item{int(it):03d}.png")
    cp[cp.item.isin(items)][["item", "prompt", "reference", "group", "level"]].to_csv(out / "items.csv", index=False)
    print(f"wrote {out}: {len(items)} photos")


def scam():
    f = ROOT / "results" / "scam_notes" / "Qwen3.5-4B" / "items.csv"
    src = DATA / "scam" / "images" / "train"
    if not f.exists() or not src.exists():
        print("scam: items or images missing, skipped")
        return
    it = read_csv(f)
    out = OUT / "scam"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in it.itertuples():
        p = src / r.file_name
        if p.exists():
            stimuli.fit_photo(str(p)).save(out / f"{r.key}.png")
            n += 1
    it.to_csv(out / "items.csv", index=False)
    print(f"wrote {out}: {n} photos")


if __name__ == "__main__":
    gqa()
    countbench()
    scam()
