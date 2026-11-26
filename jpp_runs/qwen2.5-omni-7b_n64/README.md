# Qwen2.5-Omni-7B: R-lens and J++ lens (64 prompts) and the J++ post's evals

- **Code:** `jpp_evals/` of debanshd/jspace-horizons-copy PR 17 at commit `e4e13f7`, run in the J++ harness
  `safety-research/jpp_lens` at `d7efc0d` (torch 2.11.0+cu128, transformers 5.14.1, Python 3.12.3).
- **Model:** `Qwen/Qwen2.5-Omni-7B` at `ae9e169` (the thinker, text-only prompts).
- **Hardware:** one NVIDIA H200 SXM (RunPod, US-NC-1), 25–26 Nov 2026 UTC.
- **Fit:** the harness's recipe: 8 k-means experts per layer, router on 1000 WikiText-103 prompts, Jacobian sums through the LRP
  backward pass ("rlens" rules) on 64 prompts of 128 tokens (first 16 positions skipped), source layers 0–26, target the last block.
  The R-lens is the pooled Jacobian of those sums; the J++ lens combines the experts with weights fitted on the `fit:0` split
  (67 items). Every lens is scored on the held-out split: 273 items, none of them in the fitting split.
- **Run note:** the first start filled the `jspace-media` network volume at prompt 8 of the Jacobian sums; the second start
  reused that start's finished correctness grading and router and kept the fit on the pod's local disk (`run2.log`).

| file | sha256 |
|---|---|
| `omni7b_rlens_n64.pt` | `d4e61afd18f5f82873001135d19d95a0cb77e92b0b80a605fcf243f2b43fed84` |
| `omni7b_jpp_n64.pt` | `0b2821fe0ad375e98f1ef3e4f759c8f0e45011b852899faeffe7d71bfbff703d` |
| `omni7b_jpp_out.tgz` | `bc233ccf8dbad8ec051b442263b4cdf373af623c13c2844678fc9891aa61e9d6` |

The lens files, every rank and trial, and the fit logs are in the private Hugging Face repo `debanshd/jspace-omni-jpp-runs` (commit `7639d40`), under `qwen2.5-omni-7b/`.


Wall time: correctness 57 s, router 332 s, Jacobian sums 1396 s, merge 21 s, expert weights 100 s, readout 74 s + 70 s,
probe-swap 168 s.
