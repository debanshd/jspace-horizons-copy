# The J++ post's lens evals, on the omni J-lenses and Neuronpedia baselines

These are the readout and causal evals of Ayonrinde & Lindsey, ["J++ Lens: Jacobian Filtering Enables More Faithful Workspace
Lenses"](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace) (LessWrong,
2026). They are run with the authors' harness, [`safety-research/jpp_lens`](https://github.com/safety-research/jpp_lens), at
commit `d7efc0d`. The subjects are the J-lenses in [`debanshd/jspace-omni-lenses`](https://huggingface.co/debanshd/jspace-omni-lenses),
with Neuronpedia's lenses as baselines.

None of this touches `src/`, `experiments/` or `results/`. The harness runs in its own environment, from its own `uv.lock`, so this
repository's `uv.lock` stays as it is.

## What is compared

| model | lenses | why |
|---|---|---|
| Qwen2.5-Omni-3B | omni J-lens, 250 and 500 prompts | the subject |
| Qwen2.5-Omni-7B | omni J-lens, 250 and 500 prompts | the subject |
| Qwen2.5-7B-Instruct | Neuronpedia J-lens | the text model the Omni-7B thinker was built from, same width and depth: the nearest baseline |
| Qwen3.5-4B | Neuronpedia J-lens (the one this repository reads), R-lens, J++ | the lens types side by side on one model |
| Qwen3.6-27B | Neuronpedia J-lens, J++ (an fp16 copy of the post's own lens) | calibration against the post's numbers |
| Qwen3-Omni-30B-A3B | omni J-lens, 500 prompts | only when asked for by name: it failed the halves check |

Every model also gets the logit lens. Neuronpedia fitted no lens for an omni model, so there is no same-model baseline.
`manifest.py` pins every model and lens to a commit, and every lens file to a sha256.

## The evals

- **Readout**: the five tasks multihop, multilingual, typo, association and poetry. An item is a hit when its intermediate word
  ranks in the lens's top k at any of seven layers, k · n / 8 for k = 1..7. The score is recall@1 and recall@10, averaged over
  the five tasks.
  - Multihop, multilingual and poetry items count only where the model gets the answer right itself. `run_cli.py grade-correctness`
    grades each model, because the harness ships that column for Qwen3.6-27B only.
  - Each model is scored twice: with Readout Filtering (tokens with no letter or digit taken out of the ranking, as the post
    scores J++), and over the full vocabulary (as the post scores the J-lens and R-lens).
- **Probe-swap**: 67 trials. The lens coordinate of the intermediate ("Brazil") is clamped to another word's ("Mexico") at the
  seven layers, and the trial checks whether the model's answer moves to match ("Portuguese" → "Spanish"). The score is top-1 and
  top-5 at the best of four edit scales.

## Running it

On a GPU pod with at least 80 GB of memory (an H200 to include the 30B) and a 200 GB volume, from this repository's root:

```bash
jpp_evals/run_pod.sh                       # the default five models
jpp_evals/run_pod.sh qwen3-omni-30b-a3b    # the 30B as well, separately
```

The outputs go to `/workspace/jpp/out/<model>/`: the harness's rank and recall tables, the probe-swap trials, `correctness.csv`,
a log and the wall time of each step. `summary.md` and `summary.csv` hold one row per model and lens. A stopped run resumes
where it stopped. `uv run python <repo>/jpp_evals/run_all.py --out <dir> --summary-only` (run from the checkout) rebuilds the
summary.

## Fitting an R-lens and a J++ lens for the omni models

```bash
FIT=1 jpp_evals/run_pod.sh                     # Omni-3B and Omni-7B, 64 prompts
FIT=1 FIT_PROMPTS=128 jpp_evals/run_pod.sh qwen2.5-omni-3b
```

`fit_lenses.py` runs the post's recipe in the post's harness, one stage at a time:
1. A k-means router, 8 experts per layer, on the first 1000 WikiText prompts.
2. Expert Jacobian sums through the LRP backward pass, on the first 64 prompts of 128 tokens. This is the count of the post's
   Qwen3.6-27B lens and of Neuronpedia's Qwen3.5-4B R-lens and J++ lens. Neuronpedia's J-lenses run up to 1000 prompts, but they
   stop early once the lens converges.
3. The merged experts. Their pooled Jacobian is the R-lens.
4. Expert weights fitted on the post's `fit:0` split of the readout items. The weighted experts are the J++ lens.

Every source layer is fitted, as in the omni J-lenses. Then `run_all.py --fits` scores the new lenses beside the J-lenses, and
every lens of that model is scored on the held-out split, because the J++ weights have seen the rest.

The harness's LRP surgery picks its rules by class name. `run_cli.register_omni_lrp` registers Qwen2.5-Omni's
`Qwen2_5OmniRMSNorm`, whose forward is the same as `Qwen2RMSNorm`'s, and skips the thinker's `visual` tower. Qwen3-Omni-30B's MoE
thinker is not registered, so it cannot be fitted here.

## How to read it

- **Qwen3.6-27B is the check of the setup.** It is scored on the post's held-out split, with the post's own correctness column.
  Its J++ lens should come out near the post's 55.2 recall@10 (filtered) and 59.7% probe-swap top-1. The post found Neuronpedia's
  J-lens for this model weaker than its own (35.7), so a lower number there is expected. If J++ is far from 55.2, the setup is at
  fault, not the lenses.
- **Omni-7B against Qwen2.5-7B-Instruct** compares two different models, each with its own lens. It is not a clean lens-vs-lens
  comparison.
- **Omni lens n250 against n500** is the post's fitting-data question on your own lenses.
- **The Qwen3.5-4B J++ and R-lens** were fitted by someone else (`bubskee`, see the lens's `CREDIT.md`). Their expert-weight dev
  set is unknown, so the J++ score may be partly in-sample.
- The prompts are plain text with no chat template, as in the post. The omni thinkers are instruction-tuned. The correctness
  filter drops the items they don't complete.

## Status

Nothing here has run on a GPU yet, the evals or the fits. The harness's model loading and LRP surgery were checked by reading their
code and the transformers 5.14.1 omni classes. At 64 prompts, the Jacobian sums should take roughly 10 minutes on the 3B and 40 on
the 7B on one H200. That is scaled from the omni J-lens fits (7 s per prompt on the 3B, 35 s on the 7B), not measured. The router
and the weight fit add their own time.
