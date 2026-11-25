"""The models and lenses the J++ post's evals are run on, every one pinned to a commit and a sha256.

The evals are those of Ayonrinde & Lindsey, "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses" (LessWrong,
2026), run with their harness (safety-research/jpp_lens) at JPP_LENS_COMMIT. Neuronpedia fitted no lens for an omni model, so the
baselines are Neuronpedia's lenses for the nearest text models:

- Qwen2.5-7B-Instruct: the text model Qwen2.5-Omni-7B's thinker was built from, same width (3584) and depth (28);
- Qwen3.6-27B: the post's own model, the calibration: its J-lens and J++ lens should give about the post's 35.7 and 55.2 recall@10;
- Qwen3.5-4B: the lens this repository's experiments read, with Neuronpedia's R-lens and J++ lens beside it.

Run as a script, it prints what the pod script needs: `manifest.py jpp-commit`, `manifest.py keys`, `manifest.py all-keys`.
"""

import sys
from dataclasses import dataclass

JPP_LENS_URL = "https://github.com/safety-research/jpp_lens.git"
JPP_LENS_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"

NEURONPEDIA = ("neuronpedia/jacobian-lens", "d2527c3cdbaa60dd4b65cbca1604921f044e989a")
OMNI = ("debanshd/jspace-omni-lenses", "b97bce4e54e720b36d743fe1eb9cb181187f981c")


@dataclass(frozen=True)
class Lens:
    name: str  # local file stem: the harness names a lens by it, so it must be unique within a model
    repo: tuple[str, str]  # (Hugging Face repo, commit)
    file: str
    sha256: str


@dataclass(frozen=True)
class Model:
    key: str  # run folder name
    hf_name: str
    revision: str
    n_layers: int
    lenses: tuple[Lens, ...]
    # The model class to load. None is the harness's own AutoModelForCausalLM; the omni checkpoints hold a thinker, a talker and a
    # vocoder, and only the thinker (the text-generating model the lenses were fitted on) is loaded.
    loader: str | None = None
    # "all" scores every item the model answers correctly. Qwen3.6-27B's J++ expert weights were fitted on the post's "fit:0" split,
    # so that model is scored on the post's held-out split, as the post scored it.
    items: str = "all"
    default: bool = True  # in the default run
    # (repo, commit, file) of a published fitting/held-out split ({"fit": [[eval, item], ...], "heldout": [...]}) that a lens's
    # expert weights were fitted on, checked against the items this run scores
    split_reference: tuple[str, str, str] | None = None

    @property
    def layers(self) -> list[int]:
        """the post's seven readout layers, k * n / 8 for k = 1..7 (8, 16, ..., 56 on the 64 layers of Qwen3.6-27B)"""
        return [k * self.n_layers // 8 for k in range(1, 8)]


MODELS: tuple[Model, ...] = (
    Model(
        key="qwen2.5-omni-3b",
        hf_name="Qwen/Qwen2.5-Omni-3B",
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        n_layers=36,
        loader="Qwen2_5OmniThinkerForConditionalGeneration",
        lenses=(
            Lens("omni3b_jlens_n250", OMNI, "Qwen2.5-Omni-3B/lens.pt", "8a10a569f4fe3614bf4575280079b5a2ddad7b39c173bec59f25da237992cde7"),
            Lens("omni3b_jlens_n500", OMNI, "Qwen2.5-Omni-3B/n500/lens.pt", "a7535af6f48b566e4da89fd39a54c3b418fe3ce5d001efcf34c0983bec7c141c"),
        ),
    ),
    Model(
        key="qwen2.5-omni-7b",
        hf_name="Qwen/Qwen2.5-Omni-7B",
        revision="ae9e1690543ffd5c0221dc27f79834d0294cba00",
        n_layers=28,
        loader="Qwen2_5OmniThinkerForConditionalGeneration",
        lenses=(
            Lens("omni7b_jlens_n250", OMNI, "Qwen2.5-Omni-7B/lens.pt", "d1696e876294f9e1d3510339905d64613d49125d9b74898e435a0583ef94e859"),
            Lens("omni7b_jlens_n500", OMNI, "Qwen2.5-Omni-7B/n500/lens.pt", "e2e88d60d9567a3a8225e46725938453ff7b1fc3bf899b6d710af1bf41596ba5"),
        ),
    ),
    Model(
        # Its lens failed the halves check (cosine 0.9887 < 0.99), so it is evaluated only when asked for by name.
        key="qwen3-omni-30b-a3b",
        hf_name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        revision="26291f793822fb6be9555850f06dfe95f2d7e695",
        n_layers=48,
        loader="Qwen3OmniMoeThinkerForConditionalGeneration",
        lenses=(Lens("omni30b_jlens_n500", OMNI, "Qwen3-Omni-30B-A3B-Instruct/n500/lens.pt", "257e34cc4209b9ab4e006fd41fc36d9bec7f2ba8686cbcea86ee3406d93c179e"),),
        default=False,
    ),
    Model(
        key="qwen2.5-7b-instruct",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        n_layers=28,
        lenses=(
            Lens(
                "np_qwen2.5-7b-it_jlens",
                NEURONPEDIA,
                "qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt",
                "3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29",
            ),
        ),
    ),
    Model(
        key="qwen3.5-4b",
        hf_name="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",  # as in src/jspace/config.py
        n_layers=32,
        # Its J++ weights were fitted on the harness's fit:0 split (fraction 0.2, seed 0; bubskee's provenance/fit-recipe.json),
        # 64 items, so it is scored on the held-out rest; bubskee published both lists, and the run checks them.
        items="held-out:0",
        split_reference=("bubskee/qwen3.5-4b-r-jpp-lenses", "c1b9747f3861977df31482adc142eb3b38811546", "evaluation/split.json"),
        lenses=(
            # the lens file this repository's experiments read (same sha256 as src/jspace/config.py)
            Lens(
                "np_qwen3.5-4b_jlens",
                NEURONPEDIA,
                "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
                "1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e",
            ),
            Lens("np_qwen3.5-4b_rlens", NEURONPEDIA, "qwen3.5-4b/rlens/Salesforce-wikitext/Qwen3.5-4B_rlens.pt", "843b3723b752a0345b90c46fe0f9faea4be20eefb132da198d49bc3632fa46a9"),
            Lens("np_qwen3.5-4b_jpp", NEURONPEDIA, "qwen3.5-4b/jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt", "cf0168a259a095e188b7ba17cdafceb8c92c04893a1469922a6d2a9093c1d7d7"),
        ),
    ),
    # OLMo 3 7B at 100% and 50% of its training, for J++ lenses (fit_lenses.py). The base model was trained in three stages: 5.93T
    # tokens of pretraining (stage1-step0 to step1413814), 100B of mid-training, 50B of long context (the model card), 6.08T in all.
    # 100% is the released model (main); 50%, 3.04T tokens, falls in pretraining at step 3.04 / 5.93 * 1413814 = 724788, and the
    # nearest checkpoint is stage1-step725000 (checkpoints are every 1000 steps; 4.19M tokens a step on average).
    Model(
        key="olmo-3-7b",
        hf_name="allenai/Olmo-3-1025-7B",
        revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",  # main
        n_layers=32,
        lenses=(
            # Neuronpedia's J-lens of the released model (fit_lens.py, stopped converged at 568 of 1000 prompts)
            Lens(
                "np_olmo-3-7b_jlens",
                NEURONPEDIA,
                "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt",
                "0683d9725bc9c09881c56784ee1371956ad5793b41731ea1213f025757073183",
            ),
        ),
        default=False,
    ),
    Model(
        key="olmo-3-7b-50pct",
        hf_name="allenai/Olmo-3-1025-7B",
        revision="6bfdeb0b7d9e0cf0e076529031b7576d0dab67c3",  # stage1-step725000
        n_layers=32,
        lenses=(),  # no published lens for an intermediate checkpoint: its J++ lens is scored against the logit lens only
        default=False,
    ),
    Model(
        key="qwen3.6-27b",
        hf_name="Qwen/Qwen3.6-27B",
        revision="1cda14dd71a3f47115850eb790f416f165da1cb5",
        n_layers=64,
        items="held-out:0",
        lenses=(
            Lens(
                "np_qwen3.6-27b_jlens",
                NEURONPEDIA,
                "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
                "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1",
            ),
            # Neuronpedia's fp16 copy of the post's own lens (koayon/jpp-lenses@6c96867, qwen3.6-27b/lens.pt)
            Lens("np_qwen3.6-27b_jpp", NEURONPEDIA, "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt", "bbef480b41028e156c4aa3c334ad32d1b76a4017ef3a7245eecf776bf2b58f8f"),
        ),
    ),
)


def model_by_key(key: str) -> Model:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"no model {key!r} in the manifest; the keys are {[m.key for m in MODELS]}")


def model_by_hf_name(hf_name: str) -> Model:
    found = [m for m in MODELS if m.hf_name == hf_name]
    if not found:
        raise KeyError(f"no model {hf_name!r} in the manifest: every model is loaded at a pinned revision, so add it there first")
    if len(found) > 1:  # two checkpoints of one repo: only the key says which revision to load
        raise KeyError(f"{hf_name!r} has several entries in the manifest ({[m.key for m in found]}): pass --model-key")
    return found[0]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "jpp-commit":
        print(JPP_LENS_COMMIT)
    elif what == "keys":
        print(" ".join(m.key for m in MODELS if m.default))
    elif what == "all-keys":
        print(" ".join(m.key for m in MODELS))
    else:
        sys.exit("usage: manifest.py jpp-commit | keys | all-keys")
