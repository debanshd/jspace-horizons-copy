# OLMo 3 7B: J++ lenses (64 prompts) at 100% and 50% of training, and the J++ post's evals

- **Code:** `jpp_evals/` of debanshd/jspace-horizons-copy PR 18 (PR B) at commit `3d8ccd6`, run with `FIT=1 PRUNE=0 jpp_evals/run_pod.sh
  olmo-3-7b olmo-3-7b-50pct` in the J++ harness `safety-research/jpp_lens` at `d7efc0d`, with OMP/MKL threads set to the pod's
  CPU quota (10).
- **Models:** `allenai/Olmo-3-1025-7B` at `a81bae4` (main, the released model) and `6bfdeb0` (`stage1-step725000`, 50% of
  the 6.08T training tokens).
- **Hardware:** one NVIDIA H200 SXM (RunPod, EUR-IS-4), 26 Nov 2026 UTC.
- **Fit:** 8 k-means experts per layer (router on 1000 WikiText-103 prompts), LRP ("rlens") Jacobian sums on 64 prompts of 128
  tokens, source layers 0–30, expert weights on each checkpoint's own `fit:0` split. Scored on the held-out split of each
  checkpoint's own correctness grading: 220 items (released) and 198 (50%), none in the fitting split.

| file | sha256 |
|---|---|
| `olmo7b_jpp_n64.pt` | `6b853a64ab98384058949f3349765bddc39cc05d4bb0b4283174444e77378add` |
| `olmo7b50pct_jpp_n64.pt` | `58bc90768ee0958c4398acebdbc9d38e094a212f76203e518dda34844228e233` |
| `olmo7b_jpp_out.tgz` | `3a474fdd8c16a07c19292f4cadbef60113987c458c829b78469843f589bb45e2` |

The lens files, every rank and trial, and the fit logs are in the private Hugging Face repo `debanshd/jspace-omni-jpp-runs` (commit `39fb3e3`), under `olmo-3-7b_n64/`.
Jacobian sums took 1819 s and 1811 s; the router 206 s and 208 s.
