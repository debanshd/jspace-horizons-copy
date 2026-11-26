# OLMo 3 7B: J++ lenses fitted to convergence, released (650 prompts) and 50% of training (625 prompts)

PR 18 (PR B) at `37b84e7`, `FIT=1 PRUNE=0 jpp_evals/run_pod.sh <key>` in `safety-research/jpp_lens` at `d7efc0d`, on one RTX PRO 6000
Blackwell (RunPod US-NC-3), 26 Nov 2026 UTC, OMP/MKL threads at the pod's CPU quota. Each checkpoint's Jacobian sums went in blocks of
25 prompts until Neuronpedia's rule stopped them (`<key>/convergence.csv`); each is scored on the held-out split of its own grading
(221 and 198 items, none in the fitting split). `summary.md` has the scores.

The lens files, every rank and probe-swap trial, the expert weights and the logs are in the private Hugging Face repo
`debanshd/jspace-omni-jpp-runs` under `olmo-3-7b_conv/`: `olmo-3-7b/olmo7b_jpp_n650.pt`
(sha256 `d799b024…`), `olmo-3-7b-50pct/olmo7b50pct_jpp_n625.pt` (sha256 `af4e8b55…`) and `olmo_conv_final.tgz`.
