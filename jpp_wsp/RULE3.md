# Experiment 3: the J++ post's evals on jeeva2812's OLMo 3 7B J-lens

Written 27 Nov 2026, before the run.

## The lens

`jeeva2812/olmo3-jlens-checkpoints@a661f25`, `Jall_main.pt` (sha256 `582d71ee…f2cd2f85b`): a plain Jacobian lens of
`allenai/Olmo-3-1025-7B` at `main` (the released model), one 4096 x 4096 matrix for each even layer 0 to 30. It maps into the
residual after block 30, not the last block: its layer-30 matrix is exactly the identity. The file records no prompt count. The
README says only that its Jacobians are averaged over a fixed prompt set, and the repository's 27 per-checkpoint probe files used 25
prompts of 128 tokens. Those probe files (`lenses/random_L20_*.pt`) hold 32 directions each, not a lens, so they are not scored.

`convert_jall.py` rewrites it into the released J-lens format that the harness reads (`J`, `source_layers`, `d_model`,
`n_prompts` = 0 for unknown, `provenance.target_layer` = 30), with the matrices unchanged. It runs on the pod, so the pod reads only
the public file.

## The run

The J++ post's evals (`safety-research/jpp_lens` at `d7efc0d`, PR 18's `run_all.py` at `3d8ccd6` through `score_extra.py`), on the
100% model (`olmo-3-7b`), on the `held-out:0` split from the 100% J++ fit's own `model_correctness.csv` (`inputs100/`). Scored side
by side: the logit lens, Neuronpedia's J-lens (manifest-pinned), the 100% J++ lens `olmo7b_jpp_n64` and `jeeva_Jall_main`. The
first three were scored before on this split (`olmo-3-7b_n64/out_fits/summary.md`: 33.0, 37.6 and 53.4 filtered recall@10), so
they check the setup.

## Reading, set now

Filtered macro recall@10, paired bootstrap over items within each task (2000 resamples, seed 0):
- **As good as Neuronpedia's J-lens**: jeeva's lens is within 2.0 points of it, or above it. Otherwise: below it.
- jeeva's lens against the J++ lens and the logit lens, probe-swap, and full-vocabulary recall are reported without a verdict.
- A lens that skips the last block is expected to read a little differently. That is part of what is being measured, not a defect.
