#!/usr/bin/env python
"""Download everything the experiments read, at the pinned revisions, and check it against what the original runs read.

    uv run scripts/download.py            # the data, and Qwen3.5-4B with its lens (about 11 GB)
    uv run scripts/download.py --gemma    # also Gemma-4-31B and its two lenses (69 GB more; its runs need a GPU of more than 90 GB)

The data goes to data/, and every file must have the sha256 of the file the original runs read:
    data/coco2017/   COCO 2017 val: the object crops (annotations CC BY 4.0, images under their Flickr terms)
    data/scam/       SCAM: photos with a hand-written note and their blank-note twins (CC BY-NC 4.0)
    data/gqa/        GQA testdev-balanced: natural photos and their questions
The models and lenses go to the Hugging Face cache (HF_HOME, default ~/.cache/huggingface), where the experiments load them:
each model at its pinned commit, and each lens file, which must have the sha256 of the file the original runs read.
Files already present are checked and not downloaded again; an interrupted download resumes. After this step the experiments
need no network (HF_HUB_OFFLINE=1), unless the judge has to score an answer it has no recorded verdict for.
"""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import requests
from huggingface_hub import constants, hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from jspace.config import DATA_DIR, GQA_REPO, GQA_REVISION, LENS_REPO, LENS_REVISION, MODELS, SCAM_REPO, SCAM_REVISION
from jspace.runs import load_env

DATA_SHA256 = {
    "coco2017/val2017.zip": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    "coco2017/annotations_trainval2017.zip": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    "coco2017/annotations/instances_val2017.json": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
    "scam/images.zip": "1c9d819519b8e543655f3fdcc5b0bdc03d1c5beb94600eef3f0d9924afb58b3e",
    "scam/images/train/metadata.jsonl": "a0b1d5ca360deff8c22d276fe4769cbe860203e29c922fd78fb62069c98c3375",
    "gqa/testdev_balanced_images/testdev-00000-of-00001.parquet": "ba0a57a72c483d6e4a390ecbbee05d07a55365aeac3e93f4a923ac61c9e465eb",
    "gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet": "e967085f896deee3b99b7c7a830e74fc0662a61cdbcbd0154ffbb8aea4ab67ed",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, expected, label):
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"{label}: sha256 {got} differs from the file the original runs read ({expected})")
    print(f"ok  {label}")


def fetch(url, path):
    """download a URL to a file, unless it is there"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(part, "wb") as f, tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(path)


def data():
    d = DATA_DIR
    # COCO 2017 val: the image zip is read in place; the instance annotations are taken out of their zip
    fetch("http://images.cocodataset.org/zips/val2017.zip", d / "coco2017/val2017.zip")
    check(d / "coco2017/val2017.zip", DATA_SHA256["coco2017/val2017.zip"], "data/coco2017/val2017.zip")
    if not (d / "coco2017/annotations/instances_val2017.json").exists():
        fetch("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", d / "coco2017/annotations_trainval2017.zip")
        check(d / "coco2017/annotations_trainval2017.zip", DATA_SHA256["coco2017/annotations_trainval2017.zip"], "data/coco2017/annotations_trainval2017.zip")
        zipfile.ZipFile(d / "coco2017/annotations_trainval2017.zip").extract("annotations/instances_val2017.json", d / "coco2017")
        (d / "coco2017/annotations_trainval2017.zip").unlink()
    check(d / "coco2017/annotations/instances_val2017.json", DATA_SHA256["coco2017/annotations/instances_val2017.json"], "data/coco2017/annotations/instances_val2017.json")
    # SCAM
    if not (d / "scam/images/train/metadata.jsonl").exists():
        hf_hub_download(SCAM_REPO, "images.zip", repo_type="dataset", revision=SCAM_REVISION, local_dir=d / "scam")
        check(d / "scam/images.zip", DATA_SHA256["scam/images.zip"], "data/scam/images.zip")
        zipfile.ZipFile(d / "scam/images.zip").extractall(d / "scam")
        (d / "scam/images.zip").unlink()
    check(d / "scam/images/train/metadata.jsonl", DATA_SHA256["scam/images/train/metadata.jsonl"], "data/scam/images/train/metadata.jsonl")
    # GQA testdev-balanced
    for rel in ("testdev_balanced_images/testdev-00000-of-00001.parquet", "testdev_balanced_instructions/testdev-00000-of-00001.parquet"):
        if not (d / "gqa" / rel).exists():
            hf_hub_download(GQA_REPO, rel, repo_type="dataset", revision=GQA_REVISION, local_dir=d / "gqa")
        check(d / "gqa" / rel, DATA_SHA256[f"gqa/{rel}"], f"data/gqa/{rel}")


def models(names):
    for name in names:
        spec = MODELS[name]
        print(f"{name} at {spec.revision[:8]} ({spec.size_gb:.0f} GB) ...")
        snapshot_download(name, revision=spec.revision)
        print(f"ok  {name}")
        for file, expected in ((spec.lens_file, spec.lens_sha256), (spec.jpp_file, spec.jpp_sha256)):
            if file:
                path = Path(snapshot_download(LENS_REPO, allow_patterns=[file], revision=LENS_REVISION)) / file  # the call jlens makes
                check(path, expected, f"{LENS_REPO}/{file}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemma", action="store_true", help="also Gemma-4-31B and its lenses (69 GB; only its two runs need them)")
    ap.add_argument("--data-only", action="store_true", help="only the data")
    args = ap.parse_args()
    load_env()  # HF_TOKEN from .env, if there is one
    names = [] if args.data_only else [n for n in MODELS if args.gemma or "gemma" not in n]
    cache = Path(constants.HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"data -> {DATA_DIR}; models and lenses -> {cache} ({shutil.disk_usage(cache).free / 1e9:.0f} GB free; the models need {sum(MODELS[n].size_gb for n in names):.0f} GB)")
    data()
    models(names)
    print("everything the experiments read is here and matches the original runs")


if __name__ == "__main__":
    main()
