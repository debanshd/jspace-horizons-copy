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
    lens_repo: str | None = None  # a lens from another Hub repo than LENS_REPO (None: LENS_REPO at LENS_REVISION)
    lens_revision: str | None = None  # that repo's pinned commit

    def lens_source(self):
        """(repo, revision) the lens files are read from"""
        return (self.lens_repo, self.lens_revision) if self.lens_repo else (LENS_REPO, LENS_REVISION)


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
    "Qwen/Qwen3.5-2B": ModelSpec(
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        size_gb=4.5,
        lens_file="qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
        lens_sha256="69e02a9794c340da7a69cfaa1638c0267192f25a64e7f4fa23f3021e72393c82",
        jpp_file="qwen3.5-2b/jpp/Salesforce-wikitext/Qwen3.5-2B_jpp_lens.pt",
        jpp_sha256="fbaa6ecd00a718370a658f4778c23b647a4b295ffc3f4a4836e5850784b1a649",
    ),
    "Qwen/Qwen3.5-0.8B": ModelSpec(
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        size_gb=2.0,
        lens_file="qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
        lens_sha256="aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48",
        jpp_file="qwen3.5-0.8b/jpp/Salesforce-wikitext/Qwen3.5-0.8B_jpp_lens.pt",
        jpp_sha256="39bf7690b7e1d917ccff9c0fd7418abeaa02fe334691edfb7e814ee890b99c06",
    ),
    "Qwen/Qwen3.5-9B": ModelSpec(  # added 21 Oct for the size step; Neuronpedia fitted only a J++ lens for the instruct 9B, used as its lens
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        size_gb=19.3,
        lens_file="qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt",
        lens_sha256="8cda12f7ddf11e70fcfc69a47a9574f626fe084c67807a4781f4f67106e739c5",
    ),
    "Qwen/Qwen3.5-27B": ModelSpec(  # added 21 Oct for the size step (64 layers, 16 with softmax attention)
        revision="fc05daec18b0a78c049392ed2e771dde82bdf654",
        size_gb=54.0,
        lens_file="qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt",
        lens_sha256="bb2e080ae9c9c20825adfa66c115be214ae027d867e525246a6879cb7fe6bbd6",
    ),
    "google/gemma-4-E4B-it": ModelSpec(
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        size_gb=16.0,
        lens_file="gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt",
        lens_sha256="119301db1ce48d8c5e1c9f1d374625bd8672c9522a7abb88834b1ca2a260165d",
    ),
    "google/gemma-4-E2B-it": ModelSpec(  # added for media_steer (plan.md): the small Gemma, the only other one that takes audio
        revision="4adf15748de9e1a0097b6274cee9c15836c41c37",
        size_gb=10.3,
        lens_file="gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt",
        lens_sha256="f0fbecd3de565d93fd088292b3c12b56f39a47f7ea0698a3ca5301c9690a6ab8",
    ),
    # added 24 Nov for the audio experiments: the third Gemma-4 that takes audio (48 layers, d 3840; audio embedded straight into
    # the language model). Neuronpedia has no lens for it, so the lens is hlampes/jacobian-lens's: Neuronpedia's format, every
    # source layer 0 to 46, fitted on 250 WikiText prompts, uploaded 6 Aug 2026 with no card or licence. Every commit of the
    # model has the same weights, so the lens fits the pinned one; 707f0a3b is the last before the 30 Oct tokenizer change.
    # scripts/check_lens.py tests the lens against the model before any audio run uses it.
    "google/gemma-4-12B-it": ModelSpec(
        revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        size_gb=23.9,
        lens_file="gemma-4-12B-it/WikiText/gemma-4-12B-it-lens.pt",
        lens_sha256="a8fa480c674be8d263e4899dc02b9c201ee8e95bba15262c73065fb9fc92a61a",
        lens_repo="hlampes/jacobian-lens",
        lens_revision="b23c89843f001b9cd14999f6d5a188c1c39a9146",
    ),
}

# The datasets, at the revisions the runs read; scripts/download.py checks the sha256 of every file
GQA_REPO, GQA_REVISION = "lmms-lab-encoder/GQA", "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
SCAM_REPO, SCAM_REVISION = "BLISS-e-V/SCAM", "8c674f50dfa83e1d2a0a1c95653c39c347069d72"
# VGGSound for experiments/media_steer.py (plan.md): the clips of the first shard as single mp4 files, and the label list. Both
# revisions are None until `uv run scripts/pin_media.py` resolves them on the first machine with Hugging Face access and writes
# them here; scripts/prepare_media.py stops while they are None, so no run reads an unpinned dataset.
VGG_CLIPS_REPO, VGG_CLIPS_REVISION = "harpreetsahota/VGGSound", "03f24eccd8b29562f77dc9f23e1d23ea899d8574"
VGG_CSV_REPO, VGG_CSV_REVISION = "Loie/VGGSound", "2ca3012ef85a60143a8f97b83a45bb1a7b5c2244"
# The matched video set for experiments/video_sweep.py (plan.md, part 3): 64 donor-receiver scenes on the image experiment's
# schedule, made by Nayan Saxena from Wikimedia Commons and uCO3D clips. None until scripts/pin_media.py pins it.
MATCHED_REPO, MATCHED_REVISION = "saxenan3/jspace-matched-video", "e33c6ae8ab234c5e0f9819d0ba7a36a63a198d1f"
# The matched spoken-audio set for experiments/audio_clean.py and audio_sweep.py: the same 64 scenes, each object's name spoken
# (15 recordings, one synthetic voice), by Nayan Saxena; pinned at the revision checked on 20 Nov 2026
SPOKEN_REPO, SPOKEN_REVISION = "saxenan3/jspace-matched-spoken-audio", "d95fa45effb9559e73865a20348177ceb730702e"
DATA_FILES = {
    "coco": ["coco2017/val2017.zip", "coco2017/annotations/instances_val2017.json"],
    "scam": ["scam/images/train/metadata.jsonl"],
    "gqa": ["gqa/testdev_balanced_images/testdev-00000-of-00001.parquet", "gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet"],
    "vggsound": ["vggsound/manifest.json"],
    "matched": ["matched_video/test/metadata.jsonl", "matched_video/ATTRIBUTION.tsv"],
    "spoken": ["matched_audio/metadata.jsonl", "matched_audio/checksums.json", "matched_audio/protocol.json"],
}


def require_data(data_dir, dataset):
    """stop before any model loads when a dataset is missing"""
    missing = [f for f in DATA_FILES[dataset] if not (Path(data_dir) / f).exists()]
    if missing:
        raise SystemExit(f"{dataset} data not found in {data_dir} ({', '.join(missing)}): run `uv run scripts/download.py` first")
