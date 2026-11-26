# Qwen2.5-Omni-7B: R-lens and J++ lens fitted to convergence (550 prompts), and the J++ post's evals

- **Code:** `jpp_evals/` of debanshd/jspace-horizons-copy PR 17 (PR A) at commit `04b4cd1`, run as `FIT=1 PRUNE=0 jpp_evals/run_pod.sh`
  in the J++ harness `safety-research/jpp_lens` at `d7efc0d`, with OMP/MKL threads set to the pod's CPU quota (23).
- **Model:** `Qwen/Qwen2.5-Omni-7B` at `ae9e169` (the thinker, text-only prompts).
- **Hardware:** one NVIDIA H100 SXM 80 GB (RunPod, AP-IN-2), 26 Nov 2026 UTC.
- **Fit:** as the 64-prompt run (`../qwen2.5-omni-7b/`), but the Jacobian sums go in blocks of 25 prompts until Neuronpedia's
  J-lens rule stops them: the pooled Jacobian's relative change per block, divided by √25, under 0.002, from 100 prompts on. It
  stopped at 550 prompts (0.00188); every block's change is in `fits/qwen2.5-omni-7b/omni7b_experts_conv1000.convergence.csv`.
- **Scoring:** the held-out split of this run's own correctness grading, 275 items, none in the 67-item fitting split. The grading
  differs from the 64-prompt run's (H200) on 2 of 298 graded items, so 269 of the 275 scored items are the same as that run's 273.

| file | sha256 |
|---|---|
| `omni7b_rlens_n550.pt` | `4ff3e41e22cd26bb690e20cbde55f35e8dbaa852383b7ec40a80baf1884469aa` |
| `omni7b_jpp_n550.pt` | `5f1f52162c5d3fcd8c795f23a5628936b33bd85bb56e298eeee6001c599328e0` |
| `omni7b_conv_out.tgz` | `14ce1725c86566be0d13323d49750ebd4e8e0add4ed556ef9ec6c29c8175ecbc` |

The lens files, every rank and trial, and the fit logs are in the private Hugging Face repo `debanshd/jspace-omni-jpp-runs` (commit `bedb0cf`), under `qwen2.5-omni-7b_conv550/`; `convergence.csv` here is each block's change. Jacobian sums 2.93 h over 22 blocks, router 154 s, merge 224 s, expert weights 94 s.
