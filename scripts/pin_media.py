#!/usr/bin/env python
"""Pin what experiments/media_steer.py and media_patch.py download: the two VGGSound repositories, LaSOT, and Gemma-4-E2B with its lens.

    uv run scripts/pin_media.py           # resolve today's commits and write them into src/jspace/config.py
    uv run scripts/pin_media.py --check   # exit 1 if anything is still unpinned (scripts/media_steer.sh calls this)

The revisions are resolved once, on the first machine with Hugging Face access, and committed with config.py; after that every
run reads the same bytes. The E2B lens must exist at the repository's pinned LENS_REVISION: the lens revision is not moved for
one model. Nothing that is already pinned is changed.
"""

import argparse
import re
import sys

from huggingface_hub import HfApi

from jspace import config
from jspace.runs import load_env

E2B = "google/gemma-4-E2B-it"
E2B_LENS = "gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt"


def unpinned():
    out = []
    if config.VGG_CLIPS_REVISION is None:
        out.append("VGG_CLIPS_REVISION")
    if config.VGG_CSV_REVISION is None:
        out.append("VGG_CSV_REVISION")
    if E2B not in config.MODELS:
        out.append(E2B)
    if config.LASOT_REVISION is None:
        out.append("LASOT_REVISION")
    if config.MATCHED_REVISION is None:
        out.append("MATCHED_REVISION")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="?", const="all", choices=["all", "media_steer", "media_patch", "video_sweep"], help="exit 1 if what this experiment reads is unpinned")
    args = ap.parse_args()
    todo = unpinned()
    if args.check:
        not_read = {"media_steer": {"LASOT_REVISION", "MATCHED_REVISION"}, "media_patch": {"MATCHED_REVISION"}, "video_sweep": {"LASOT_REVISION"}}
        todo = [t for t in todo if t not in not_read.get(args.check, set())]  # each experiment checks only what it reads
        if todo:
            print("unpinned: " + ", ".join(todo))
            sys.exit(1)
        print("everything media_steer reads is pinned")
        return
    if not todo:
        print("everything media_steer reads is pinned already")
        return
    load_env()
    api = HfApi()
    path = config.ROOT / "src/jspace/config.py"
    src = path.read_text()
    if config.VGG_CLIPS_REVISION is None:
        sha = api.dataset_info(config.VGG_CLIPS_REPO).sha
        src = src.replace(f'VGG_CLIPS_REPO, VGG_CLIPS_REVISION = "{config.VGG_CLIPS_REPO}", None', f'VGG_CLIPS_REPO, VGG_CLIPS_REVISION = "{config.VGG_CLIPS_REPO}", "{sha}"')
        print(f"{config.VGG_CLIPS_REPO} at {sha}")
    if config.VGG_CSV_REVISION is None:
        sha = api.dataset_info(config.VGG_CSV_REPO).sha
        src = src.replace(f'VGG_CSV_REPO, VGG_CSV_REVISION = "{config.VGG_CSV_REPO}", None', f'VGG_CSV_REPO, VGG_CSV_REVISION = "{config.VGG_CSV_REPO}", "{sha}"')
        print(f"{config.VGG_CSV_REPO} at {sha}")
    if config.MATCHED_REVISION is None:
        sha = api.dataset_info(config.MATCHED_REPO).sha
        src = src.replace(f'MATCHED_REPO, MATCHED_REVISION = "{config.MATCHED_REPO}", None', f'MATCHED_REPO, MATCHED_REVISION = "{config.MATCHED_REPO}", "{sha}"')
        print(f"{config.MATCHED_REPO} at {sha}")
    if config.LASOT_REVISION is None:
        sha = api.dataset_info(config.LASOT_REPO).sha
        src = src.replace(f'LASOT_REPO, LASOT_REVISION = "{config.LASOT_REPO}", None', f'LASOT_REPO, LASOT_REVISION = "{config.LASOT_REPO}", "{sha}"')
        print(f"{config.LASOT_REPO} at {sha}")
    if E2B not in config.MODELS:
        info = api.model_info(E2B, files_metadata=True)
        size_gb = sum((s.size or 0) for s in info.siblings) / 1e9
        lens = api.get_paths_info(config.LENS_REPO, [E2B_LENS], revision=config.LENS_REVISION)
        if not lens or lens[0].lfs is None:
            raise SystemExit(f"{E2B_LENS} is not in {config.LENS_REPO} at the pinned revision {config.LENS_REVISION[:8]}: E2B cannot be pinned without moving the lens revision")
        block = (
            f'    "{E2B}": ModelSpec(  # added for media_steer (plan.md): the small Gemma, the only other one that takes audio\n'
            f'        revision="{info.sha}",\n'
            f"        size_gb={size_gb:.1f},\n"
            f'        lens_file="{E2B_LENS}",\n'
            f'        lens_sha256="{lens[0].lfs.sha256}",\n'
            "    ),\n"
        )
        anchor = re.search(r'    "google/gemma-4-E4B-it": ModelSpec\(.*?\n    \),\n', src, re.S)
        assert anchor, "the Gemma-4-E4B entry was not found in config.py"
        src = src[: anchor.end()] + block + src[anchor.end() :]
        print(f"{E2B} at {info.sha}, lens sha256 {lens[0].lfs.sha256}")
    path.write_text(src)
    print(f"written to {path}: commit it, so every later run reads the same revisions")


if __name__ == "__main__":
    main()
