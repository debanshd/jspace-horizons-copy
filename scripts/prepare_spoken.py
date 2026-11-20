#!/usr/bin/env python
"""Download the matched spoken-audio set (saxenan3/jspace-matched-spoken-audio) that experiments/audio_clean.py and
audio_sweep.py read, and check it.

    uv run scripts/prepare_spoken.py

The dataset is downloaded at its pinned revision (config.SPOKEN_REVISION) into data/matched_audio/. The script stops unless:
every file listed in checksums.json has that sha256; every scene's recordings carry the sha256 metadata.jsonl gives them and
decode as 3-second 16 kHz mono PCM16; the questions in protocol.json are spoken.QUESTIONS word for word; and the scenes, labels
and set-question names are the matched video set's, scene by scene, when that set is present (the audio set is built on it).
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from jspace import matched, spoken
from jspace.config import DATA_DIR, SPOKEN_REPO, SPOKEN_REVISION
from jspace.runs import load_env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()
    load_env()
    root = Path(snapshot_download(SPOKEN_REPO, repo_type="dataset", revision=SPOKEN_REVISION, local_dir=f"{args.data_dir}/{spoken.DIR}"))
    sums = json.loads((root / "checksums.json").read_text())
    bad = [f for f, h in sorted(sums.items()) if spoken.sha256(root / f) != h]
    S = spoken.scenes(args.data_dir)
    recs = spoken.recordings(S)
    for _label, rel, h in sorted(recs.values()):
        if spoken.sha256(root / rel) != h:
            bad.append(rel)
        spoken.load_wav(root / rel)  # asserts the rate, the channels and the length
    proto = json.loads((root / "protocol.json").read_text())["questions"]
    if proto != spoken.QUESTIONS:
        raise SystemExit(f"protocol.json's questions differ from spoken.QUESTIONS: {proto}")
    labels = {r[f"{role}_label"] for r in S for role in ("donor", "receiver")} | {w for r in S for w in r["membership"]}
    unknown = sorted(labels - set(matched.VOCAB))
    if bad or unknown or len(S) != 64:
        raise SystemExit(f"{len(S)} scenes; files differing from their sha256 {bad[:4]}; labels outside the vocabulary {unknown}")
    if (Path(args.data_dir) / matched.DIR / "test/metadata.jsonl").exists():
        V = matched.scenes(args.data_dir)
        key = lambda r: (r["scene_id"], r["donor_label"], r["receiver_label"], list(r["membership"]))  # noqa: E731
        if [key(r) for r in S] != [key(r) for r in V]:
            raise SystemExit("the spoken scenes differ from the matched video set's")
    print(f"ok  {len(S)} scenes, {len(recs)} recordings, every file as checksums.json records it, at {SPOKEN_REVISION[:8]}")


if __name__ == "__main__":
    main()
