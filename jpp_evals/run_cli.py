"""The jpp_lens command line (scripts/jpp_cli.py) with two changes, and one added stage.

- Every model is loaded at the revision manifest.py pins (the harness loads the hub's main branch).
- The omni models load their thinker class: the harness loads AutoModelForCausalLM, which has no mapping for an omni checkpoint.
  The thinker has the layout jlens.from_hf looks for first (model.layers, model.norm, model.embed_tokens, lm_head), and its
  text-only forward pass uses plain positions (the rope's three sections all get the token index), as when the lenses were fitted.
- `grade-correctness`: which readout items the model answers itself, the column the readout evals filter on. The harness ships the
  column for Qwen3.6-27B only.

Run from the root of a jpp_lens checkout at manifest.JPP_LENS_COMMIT, in its environment (`uv run python <this file> <stage> ...`):
the harness reads its eval data by relative path.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))  # jpp_cli.py lives in the checkout's scripts/, not in an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jlens  # noqa: E402
import jpp_cli  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS, READOUT_EVALS_DATA_DIR, load_readout_eval_items  # noqa: E402
from manifest import model_by_hf_name  # noqa: E402


def load_model(args: argparse.Namespace):
    """The --hf-model-name model in bf16 at its pinned revision, as the harness's LensModel (replaces jpp_cli._load_model)"""
    spec = model_by_hf_name(args.hf_model_name)
    cls = getattr(transformers, spec.loader) if spec.loader else transformers.AutoModelForCausalLM
    hf_model = cls.from_pretrained(
        spec.hf_name,
        revision=spec.revision,
        dtype=torch.bfloat16,
        attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        device_map=getattr(args, "device_map", None),
    )
    if getattr(args, "device_map", None) is None:
        hf_model = hf_model.cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec.hf_name, revision=spec.revision)
    model = jlens.from_hf(hf_model, tokenizer)
    # The lens checks compare these with the lens file's d_model; a mismatch here means the wrong sub-model was found.
    assert (model.n_layers, len(model.layers)) == (spec.n_layers, spec.n_layers), (model.n_layers, len(model.layers), spec.n_layers)
    return model


def grade_correctness(argv: list[str]) -> None:
    """One greedy step per item of the five macro evals; writes eval, item and a <hf-model-name> column (True, False, or empty for
    the ungradeable typo and association items), the format of the harness's model_correctness.csv"""
    p = argparse.ArgumentParser(prog="run_cli.py grade-correctness")
    p.add_argument("--hf-model-name", required=True)
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--device-map", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    spec = model_by_hf_name(args.hf_model_name)
    model = load_model(args)
    items = load_readout_eval_items(READOUT_EVALS_DATA_DIR, slugs=MACRO_EVALS)
    runner = jpp_cli.readout_runner(model, layers=spec.layers, max_seq_len=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN)
    df = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    graded = df[args.hf_model_name].dropna()
    print(f"{args.hf_model_name}: {int(graded.sum())} of {len(graded)} gradeable items correct; {len(df) - len(graded)} ungradeable, kept")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "grade-correctness":
        grade_correctness(argv[1:])
        return 0
    jpp_cli._load_model = load_model  # the one place the harness loads a model
    return jpp_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
