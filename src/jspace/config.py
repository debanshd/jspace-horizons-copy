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


# Models whose J-lens this project fits itself (scripts/fit_lens.py): audio-trained models no published lens covers. Each is
# loaded as its text-generating part ("thinker", the class named in loader), which reads the audio, pictures and video and
# writes the text; the speech-output parts are not loaded. Pinned at the commits checked on 25 Nov 2026.
@dataclass(frozen=True)
class FitSpec:
    revision: str
    loader: str  # the transformers class that loads the model's text-generating part from the full checkpoint
    size_gb: float  # the full checkpoint's download size
    n_layers: int
    d_model: int


FIT_MODELS = {
    "Qwen/Qwen2.5-Omni-3B": FitSpec("f75b40e3da2003cdd6e1829b1f420ca70797c34e", "Qwen2_5OmniThinkerForConditionalGeneration", 12.0, 36, 2048),
    "Qwen/Qwen2.5-Omni-7B": FitSpec("ae9e1690543ffd5c0221dc27f79834d0294cba00", "Qwen2_5OmniThinkerForConditionalGeneration", 22.4, 28, 3584),
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": FitSpec("26291f793822fb6be9555850f06dfe95f2d7e695", "Qwen3OmniMoeThinkerForConditionalGeneration", 70.5, 48, 2048),
    # added 25 Nov: the mid-size Gemma-4 for the video experiments (pictures and video, no audio): a mixture of experts, 128
    # experts with 8 active (about 4B parameters per token); no published lens covers it. Loaded whole, as Session loads Gemma.
    "google/gemma-4-26B-A4B-it": FitSpec("4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "AutoModelForMultimodalLM", 51.6, 30, 2816),
    # added 25 Nov: the dense mid-size Gemma-4 (pictures, video and audio). Its two published lenses failed check_lens (hlampes,
    # 250 prompts; lamm-mit, 25 layers mapped to layer 46), and diagnose_lens found its late layers unreadable by either. Fitted
    # here with this recipe, at the revision config.MODELS pins for it, to tell a lens problem from the model's own.
    "google/gemma-4-12B-it": FitSpec("707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "AutoModelForMultimodalLM", 23.9, 48, 3840),
    # added 25 Nov: not to be used, only refitted, to check this recipe against Neuronpedia's own lens of it (LENS_CHECKS)
    "Qwen/Qwen3.5-2B": FitSpec("15852e8c16360a2fea060d615a32b45270f8a8fc", "AutoModelForImageTextToText", 4.5, 24, 2048),
    # added 25 Nov: the mid-size Qwen for the video experiments (PR #15). Neuronpedia fitted only a J++ lens for the instruct 9B,
    # and the third-party J-lens (bcywinski) failed check_lens and stays a miss; the experiments want a J-lens, so it is fitted
    # here, at the revision config.MODELS pins, and validated by LENS_CHECKS like the omni lenses (250 prompts, then 500).
    "Qwen/Qwen3.5-9B": FitSpec("c202236235762e1c871ad0ccb60c8ee5ba337b9a", "AutoModelForImageTextToText", 19.3, 32, 4096),
    # added 25 Nov: not to be used, only refitted, for the Gemma family's check 1 (LENS_FAMILIES): the base Gemma-4-E2B, the model
    # Neuronpedia fitted its E2B lens on (455 prompts, 12 Jun 2026), at its last commit before that fit (3 Jun; a tokenizer fix
    # came on 3 Nov)
    "google/gemma-4-E2B": FitSpec("63db66a33dc06d58c02b1e887446e103c202602c", "AutoModelForMultimodalLM", 10.2, 35, 1536),
}

# How a lens fitted here is validated (written 25 Nov, before any of these checks ran). Agreement with the model's next-token
# prediction is not a test of a J-lens: the J-lens paper finds its lens further from the model's output than the plain logit
# lens through most layers, and Neuronpedia's own Qwen lenses fail scripts/check_lens.py. The checks, each over the layers the
# experiments read (from 0.3 of the depth, Session.layers):
# 1. reproduce: this recipe (scripts/fit_lens.py) refits a lens Neuronpedia fitted, Qwen3.5-2B's (283 prompts, its config.yaml),
#    with as many prompts, and the two agree: the cosine of their maps (each layer's J, flattened) is at least MIN_COSINE at every
#    such layer (scripts/compare_lenses.py). If not, no lens fitted here is used, and nothing more is fitted until the
#    difference is understood.
# 2. stable: the lens's two halves (its prompts 0-499 and 500-999, fitted apart and merged into it) agree by the same measure.
# 3. reads concepts (revised 25 Nov, before any check ran: the first wording compared raw scores, which a lens with no Jacobian,
#    the identity map, could pass): on the paper's lens-quality prompts (five sets, src/jspace/lens_eval), the lens's margin, its
#    pass@10 averaged over the sets minus its own model's logit lens's (the identity map's) on the same prompts, is at least
#    the lowest margin of Neuronpedia's Qwen3.5 J-lenses (READOUT_CALIBRATION), measured the same way (scripts/lens_readout.py).
#    The check is informative only if that lowest margin is above 0, i.e. every calibration lens reads concepts better than
#    its logit lens, so the identity map (margin 0) fails it. If not, check 3 is void: no lens fitted here can be shown to
#    read concepts beyond the logit lens, and none is used.
# A lens is used only if check 1 passed and it passes 2 and 3 (scripts/validate_lens.py). Qwen lenses only: a Gemma lens needs
# its own reproduction and calibration (Neuronpedia's Gemma-4 lenses are fitted on the base models, config.MODELS pins them for
# the instruct ones), written before any Gemma lens is validated.
#
# What ran (runs/lens_repro, runs/lens_readout_np, 25 Nov): check 1 passed (cosine >= 0.992 at every read layer), and check 3
# came out void: only the 2B's lens beats its logit lens (+0.102); the 4B's (-0.033) and the 27B's (-0.025), which the video
# results read, do not. By the rule above, no fitted lens would be used.
#
# Changed 25 Nov AFTER that result, by the project owner's explicit approval ("I approve using the 500-prompt lenses"), and before
# any audio result was read through a lens fitted here. This is a change of rule after a result, recorded as such. The reason:
# the audio experiments are to repeat the video ones, lens parts included, so a fitted lens is put on the footing of the video's
# lenses, which check 3 fails too and whose prompt counts are where Neuronpedia's fits stopped changing (233 to 861; the 4B's
# 1000). So:
# - check 3 is reported beside each lens, as a caveat that applies to the video's Qwen lenses alike, and decides nothing;
# - check 2 asks that the lens has stopped changing: its prompts 0-249 (the 250-prompt fit) and 250-499, fitted apart and merged
#   into a 500-prompt lens, agree by check 1's measure (cosine >= MIN_COSINE at every read layer);
# - a lens is used if check 1 passed and it passes check 2. The deciding lens is the 500-prompt one (no longer 1000 prompts).
LENS_CHECKS = dict(reproduce="Qwen/Qwen3.5-2B", reproduce_n=283, min_cosine=0.99, min_depth=0.3, halves=(250, 500), k=10, readout_decides=False)
READOUT_CALIBRATION = (
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-27B",
)  # the J-lenses config.MODELS pins (Neuronpedia's repo; the 4B's is mntss's 1000-prompt fit), which the original results read
# The checks per model family (written 25 Nov; for the Gemma family before any Gemma check ran). Each family has its own check 1
# (a Neuronpedia lens of that family refitted with as many prompts) and its own check 3 calibration (the Neuronpedia J-lenses of
# that family the video results read, as config.MODELS pins them). The thresholds, the halves and check 3's standing (a caveat,
# readout_decides) are LENS_CHECKS's for both. Gemma's check 1 refits the base Gemma-4-E2B, because Neuronpedia fitted its Gemma-4
# lenses on the base models; config.MODELS pins that lens file for the instruct E2B (REPRO_PINNED).
LENS_FAMILIES = {
    "Qwen": dict(reproduce=LENS_CHECKS["reproduce"], reproduce_n=LENS_CHECKS["reproduce_n"], calibration=READOUT_CALIBRATION),
    "google": dict(reproduce="google/gemma-4-E2B", reproduce_n=455, calibration=("google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-31B-it")),
}
REPRO_PINNED = {"google/gemma-4-E2B": "google/gemma-4-E2B-it"}  # a refitted model -> the config.MODELS entry pinning its Neuronpedia lens


def lens_family(name):
    """a model's LENS_FAMILIES key: its Hub owner (Qwen, google)"""
    fam = name.split("/")[0]
    if fam not in LENS_FAMILIES:
        raise SystemExit(f"{name}: no lens checks for the {fam} family (config.LENS_FAMILIES)")
    return fam


LENS_EVAL_FILES = {  # src/jspace/lens_eval/, from anthropics/jacobian-lens at 581d398 (NOTICE.md)
    "association": "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
    "multihop": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
    "multilingual": "fa70b9bd89416a6d8d985a80dc628b109ae6fd3b25b9275c0fc5065d7ff4a0ef",
    "poetry": "6aeb3415c5a5c3f3827c9efe63f006de02f5ef39a816bbac68e15e733aba60cc",
    "typo": "9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa",
}
# The fitting corpus: WikiText-103 (raw), Neuronpedia's corpus for its lenses, at the revision lamm-mit's Gemma-4 lenses record;
# the first training shard and the validation split (for the fitted lens's report), each checked against its sha256
WIKITEXT_REPO, WIKITEXT_REVISION = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES = {
    "train": ("wikitext-103-raw-v1/train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
    "validation": ("wikitext-103-raw-v1/validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"),
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
