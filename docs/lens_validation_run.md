# Validating the lenses fitted here: how to run it

For whoever runs the pod. The three checks and their thresholds are written in `src/jspace/config.py` (`LENS_CHECKS`), before
any of them ran. **Do not edit a check, a threshold or a script to make a step pass.** If a step fails, report it.

1. **Reproduce.** Our fitting code refits Neuronpedia's Qwen3.5-2B lens with as many prompts (283). The two must agree: cosine
   ≥ 0.99 at every layer the experiments read.
2. **Stable.** A lens's two halves (prompts 0–249 and 250–499) agree by the same measure.
3. **Reads concepts beyond the logit lens.** On the J-lens paper's concept prompts, the lens's margin over its own model's
   logit lens, against the smallest margin among Neuronpedia's Qwen3.5 2B, 4B and 27B J-lenses.

**Changed on 25 Nov, after check 3 came out void, with the project owner's explicit approval** (`config.LENS_CHECKS`). Check 3
is now a caveat: it is reported and decides nothing, because the video's Qwen lenses fail it too. A lens is used if check 1
passed and it passes check 2. The deciding lens has 500 prompts, not 1,000, as Neuronpedia's fits stopped at 233–861 prompts.

## 1. Done (25 Nov): check 1 and check 3's calibration

Check 1 passed: cosine ≥ 0.992 at every read layer. Check 3 came out void: the margins were 2B +0.102, 4B −0.033 and 27B
−0.025. Kept for the record:

In the worktree at PR #12's head (`claude/omni-lenses`):

```bash
CHECK_RECIPE=1 scripts/fit_lenses.sh 2>&1 | tee runs/check_recipe.log
```

It downloads Qwen3.5-2B, 4B and 27B (about 68 GB), fits the 2B with 283 prompts into `runs/lens_repro_fit`, compares that fit
with Neuronpedia's lens into `runs/lens_repro`, and scores the three Neuronpedia lenses' concept readout into
`runs/lens_readout_np`. Then stop the pod.

Report:

- `runs/lens_repro/Qwen3.5-2B/compare.json`: `passes`, `lowest`, and `cosine` and `cosine_departure` by layer;
- for each of the three models, from `runs/lens_readout_np/<model>/summary.json`: `score`, and `logit_pass@10` → `mean`.
- the last line of `runs/check_recipe.log`: check 3's margins, its bar, and whether it is informative or void.

If check 1 fails, nothing more is fitted. The difference then has to be understood first. Neuronpedia fitted with a later jlens
commit (`4430656`, ours is `581d398`), transformers 5.17, and its own prompt order, so report those beside the numbers.

## 2. NOW: the 500-prompt lenses (about 3 hours on one H200, about $17)

Run it in the worktree that holds `runs/fit_lens` (the three 250-prompt fits) and `runs/lens_repro` (step 1):

```bash
EXTEND_TO=500 PUSH_TO=debanshd/jspace-omni-lenses scripts/fit_lenses.sh 2>&1 | tee runs/fit_extend500.log
```

Per model it does the following:

1. Fits prompts 250–499 (`runs/fit_lens_ext500`).
2. Merges them with the 250-prompt fit into the 500-prompt lens (`runs/fit_lens_n500`). The lens is uploaded under
   `<model>/n500/`.
3. Compares the two halves (check 2, `runs/lens_halves`).
4. Scores the concept readout (check 3, a caveat, `runs/lens_readout`).
5. Writes the verdict to `runs/lens_validation/<model>/summary.json`.

Report:

- each verdict file whole;
- the Hugging Face repo's new commit;
- each `runs/fit_lens_n500/<model>/meta.json`: `n_prompts` and `lens_sha256`.

Then stop the pod.
