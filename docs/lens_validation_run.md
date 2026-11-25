# Validating the lenses fitted here: how to run it

For whoever runs the pod. The three checks and their thresholds are written in `src/jspace/config.py` (`LENS_CHECKS`), before
any of them ran. **Do not edit a check, a threshold or a script to make a step pass.** If a step fails, report it.

1. **Reproduce.** Our fitting code refits Neuronpedia's Qwen3.5-2B lens with as many prompts (283). The two must agree: cosine
   ≥ 0.99 at every layer the experiments read.
2. **Stable.** A lens's two halves (prompts 0–499 and 500–999) agree by the same measure.
3. **Reads concepts.** On the J-lens paper's concept prompts, the lens scores at least as well as the weakest of Neuronpedia's
   Qwen3.5 2B, 4B and 27B J-lenses.

A lens is used only if check 1 passed and it passes checks 2 and 3 (`scripts/validate_lens.py`).

## 1. NOW: check 1 and check 3's calibration (one H200, about 1 hour, about $4)

In a worktree at PR #12's head (`claude/omni-lenses`) or PR #14's (`claude/omni-models`, which includes it):

```bash
CHECK_RECIPE=1 scripts/fit_lenses.sh 2>&1 | tee runs/check_recipe.log
```

It downloads Qwen3.5-2B, 4B and 27B (about 68 GB), fits the 2B with 283 prompts into `runs/lens_repro_fit`, compares that fit
with Neuronpedia's lens into `runs/lens_repro`, and scores the three Neuronpedia lenses' concept readout into
`runs/lens_readout_np`. Then stop the pod.

Report:

- `runs/lens_repro/Qwen3.5-2B/compare.json`: `passes`, `lowest`, and `cosine` and `cosine_departure` by layer;
- for each of the three models, from `runs/lens_readout_np/<model>/summary.json`: `score`, and `logit_pass@10` → `mean`.

If check 1 fails, nothing more is fitted. The difference then has to be understood first. Neuronpedia fitted with a later jlens
commit (`4430656`, ours is `581d398`), transformers 5.17, and its own prompt order, so report those beside the numbers.

## 2. Later, with a separate go-ahead: the 1,000-prompt lenses (one H200, about 9.4 hours, about $50)

Only if check 1 passed. Run it in the worktree that holds `runs/fit_lens`, the three 250-prompt fits:

```bash
EXTEND_TO=1000 PUSH_TO=debanshd/jspace-omni-lenses scripts/fit_lenses.sh 2>&1 | tee runs/fit_extend.log
```

Per model it does the following:

1. Fits prompts 250–499 and 500–999 separately.
2. Merges the three parts into the 1,000-prompt lens. The lens is uploaded under `<model>/n1000/`.
3. Compares the halves (check 2).
4. Scores the concept readout (check 3).
5. Writes the verdict to `runs/lens_validation/<model>/summary.json`.

Report:

- each verdict file whole;
- the Hugging Face repo's new commit;
- each `runs/fit_lens_n1000/<model>/meta.json`: `n_prompts` and `lens_sha256`.
