#!/usr/bin/env python
"""Download the matched video set (saxenan3/jspace-matched-video) that experiments/video_sweep.py reads, and check every clip.

    uv run scripts/prepare_matched.py

The dataset is downloaded at its pinned revision (config.MATCHED_REVISION; `uv run scripts/pin_media.py` pins it) into
data/matched_video/. Every mp4 named in test/metadata.jsonl must have the sha256 that ATTRIBUTION.tsv records for it, and
every scene's labels must be in stimuli.VOCAB; the script stops otherwise. Nothing is decoded here: video_sweep decodes the
frames each model's sampler picks.
"""

import argparse

from huggingface_hub import snapshot_download

from jspace import matched
from jspace.config import DATA_DIR, MATCHED_REPO, MATCHED_REVISION
from jspace.runs import load_env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()
    load_env()
    if MATCHED_REVISION is None:
        raise SystemExit("the matched video set is not pinned in src/jspace/config.py: run `uv run scripts/pin_media.py` first")
    root = snapshot_download(MATCHED_REPO, repo_type="dataset", revision=MATCHED_REVISION, local_dir=f"{args.data_dir}/{matched.DIR}")
    S, A = matched.scenes(args.data_dir), matched.attribution(args.data_dir)
    files = {r[f"{role}_clip_id"]: r[f"{role}_file_name"] for r in S for role in ("donor", "receiver")}
    bad = []
    for cid, name in sorted(files.items()):
        got = matched.sha256(f"{root}/test/{name}")
        if cid not in A or A[cid]["sha256"] != got:
            bad.append((cid, got[:12], A.get(cid, {}).get("sha256", "not in ATTRIBUTION.tsv")[:12]))
    labels = {r[f"{role}_label"] for r in S for role in ("donor", "receiver")} | {w for r in S for w in r["membership"]}
    unknown = sorted(labels - set(matched.VOCAB))
    if bad or unknown:
        raise SystemExit(f"{len(bad)} clips differ from ATTRIBUTION.tsv {bad[:3]}; labels outside the vocabulary {unknown}")
    print(f"ok  {len(S)} scenes, {len(files)} clips, every sha256 as ATTRIBUTION.tsv records it, at {MATCHED_REVISION[:8]}")


if __name__ == "__main__":
    main()
