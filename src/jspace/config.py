"""Pinned revisions and default paths: everything a run downloads is fixed here, so a re-run reads the same bytes."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ModelSpec:
    revision: str  # Hugging Face commit of the model
    size_gb: float  # download size
    lens_file: str  # J-lens file in LENS_REPO
    lens_sha256: str  # of the lens file the original runs read
    jpp_file: str | None = None  # J++ lens file, where Neuronpedia fitted one
    jpp_sha256: str | None = None


LENS_REPO = "neuronpedia/jacobian-lens"
# The original runs read this repo's main branch on 13-14 Oct 2026 (commit 9fa267f, since rewritten out of the hub's history).
# The lens files are unchanged at this commit: the Qwen3.5-4B lens has the sha256 of the file the runs read, and the Gemma files
# were last changed before the runs. scripts/download.py checks every lens file against these hashes.
LENS_REVISION = "e166deb5ad540932100ef803c8c9bcc2a5453694"

MODELS = {
    "Qwen/Qwen3.5-4B": ModelSpec(
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        size_gb=9.3,
        lens_file="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        lens_sha256="1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e",
    ),
    "google/gemma-4-31B-it": ModelSpec(
        revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
        size_gb=62.6,
        lens_file="gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt",
        lens_sha256="d78d3e1483f1f2159dc955ca2871d796fd995fc4613d85103ad04475b810415f",
        jpp_file="gemma-4-31b/jpp/Salesforce-wikitext/gemma-4-31B_jpp_lens.pt",
        jpp_sha256="ef50bc54b04738e558b7d86847889bb5d60437e45276b1952e73da91b4e9f5fc",
    ),
}

# The datasets, at the revisions the runs read; scripts/download.py checks the sha256 of every file
GQA_REPO, GQA_REVISION = "lmms-lab-encoder/GQA", "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
SCAM_REPO, SCAM_REVISION = "BLISS-e-V/SCAM", "8c674f50dfa83e1d2a0a1c95653c39c347069d72"
DATA_FILES = {
    "coco": ["coco2017/val2017.zip", "coco2017/annotations/instances_val2017.json"],
    "scam": ["scam/images/train/metadata.jsonl"],
    "gqa": ["gqa/testdev_balanced_images/testdev-00000-of-00001.parquet", "gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet"],
}


def require_data(data_dir, dataset):
    """stop before any model loads when a dataset is missing"""
    missing = [f for f in DATA_FILES[dataset] if not (Path(data_dir) / f).exists()]
    if missing:
        raise SystemExit(f"{dataset} data not found in {data_dir} ({', '.join(missing)}): run `uv run scripts/download.py` first")
