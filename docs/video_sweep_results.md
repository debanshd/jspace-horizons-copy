# video_sweep and media_patch: results so far

PR #4's experiments, run on 18 November 2026 with `scripts/pod_run.sh` (PR #4 at `e9b2afe`, matched video set pinned at
`e33c6ae8`). The rules and predictions are PR #4's, written before the runs. Every number is copied from a run's
`summary.json`; intervals are 95% bootstrap intervals. The run folders are in `docs/video_sweep_partial.tar.gz`
(5.2 MB, sha256 `fdbde2b488a7aa91c6b060389f3bca8e472b1b997c11af3a8f0ce0fa3d898296`).

This is a partial record: the runs below are the ones finished by 05:04 UTC. The rest are still running (see the end).

## Where the runs are

| model | pod | smoke run twice | full `video_sweep` (64 scenes) | `media_patch` |
|---|---|---|---|---|
| Qwen3.5-0.8B | L40S | identical (5,700 rows) | **done** (66 min) | after the four `video_sweep` runs |
| Gemma-4-E2B | L40S | identical (6,570 rows) | running since 04:31 UTC | after the four |
| Qwen3.5-4B | L40S | to come | to come | to come |
| Gemma-4-E4B | L40S | to come | to come | to come |
| Qwen3.5-27B | H200 | identical (9,750 rows) | **not run** (budget: a full run is about 6.5 h at $5.29/hr) | running since 05:04 UTC |
| Gemma-4-31B | — | not run | not run | not run |

The smoke runs (5 scenes) give counts only, no verdicts. The first pair of Qwen3.5-0.8B smoke runs differed in 1 of
5,700 answers, because each engine re-measured the free GPU memory and the batch sizes moved; after PR #4's `e4d3f85` (one
token budget per run) every pair is identical. The first pair is kept in the archive under `runs/attempt_budget/`.

## Qwen3.5-0.8B, full run (`runs/video_sweep/Qwen3.5-0.8B`)

64 scenes; Qwen3.5's own sampler gives 4 to 90 frames per scene (2 per second), so prompts run from 588 to 11,700
media tokens on the moving and frozen inputs, and 294 on the still picture. Patch windows start at layers −1, 3, 7, 11,
15, 19 and 23.

**Gates (clean accuracy).** The naming question fails its 0.90 gate on every input, so every patching and steering verdict
is "not perceived, untestable", and every prediction (P1–P5, and #3's 1–8) is untestable.

| input | say | animal | letter | set | ride |
|---|---|---|---|---|---|
| moving | **0.83** | 1.00 | 0.47 | 0.75 | 0.61 |
| frozen | **0.78** | 0.94 | 0.38 | 0.70 | 0.66 |
| still | **0.83** | **0.00** | 0.17 | 0.00 | 0.00 |

**Controls.** Self-patching gives the clean answer on 0.97–0.98 of rows (the machinery works). The every-layer donor patch
makes the name follow the donor on 0.84 [0.75, 0.92], under the 0.9 the positive control needs.

**Horizons, reported though the gates fail** (the last window start at which a question follows the donor on ≥ 0.5 of
scenes): say 15 on all three inputs; animal, set and ride 3 on the moving and frozen inputs; letter none.

**The still input's zeros are a prompt problem, not a finding.** On the still picture the animal question's clean answer
is "The object in image…" on all 64 scenes, not a digit, so it scores 0. The answer runs past the tokens the scoring
reads. The set and ride questions show the same zeros. This needs a look at the still prompt before any still-input result
is read. On the moving and frozen inputs the animal answers are digits (all correct).

**Steering and the other interventions** (say and animal, dose 2.0 calibrated): untestable, as naming fails its gate.
Compared block by block, the moving video differs from the frozen one in the middle block (0.08 [0.02, 0.17]), and from
the still picture in the early and middle blocks.

**Run.** 3,959 s; mean GPU utilization 82.4%, median 100%; the encoder cache was on for the moving and frozen inputs and
off for the still one; all 64 scenes fit one chunk of donor states.

## Still running

- L40S (`98xlmtskoctitm`): Gemma-4-E2B's full run, then Qwen3.5-4B and Gemma-4-E4B (smoke runs twice, then 64 scenes),
  then `media_patch` on the four. The chain logs a smoke difference or a failed run and goes on, and the pod stops itself at
  the end.
- H200 (`td2g5zgd71uj8o`): `media_patch` on Qwen3.5-27B, then the pod stops itself.
