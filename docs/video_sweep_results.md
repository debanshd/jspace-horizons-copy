# video_sweep and media_patch: results

PR #4's experiments, run on 18 November 2026 with `scripts/pod_run.sh` (PR #4 at `e9b2afe`, matched video set pinned at
`e33c6ae8`). The rules and predictions are PR #4's, written before the runs. Every number below is copied from a run's
`summary.json`; intervals are 95% bootstrap intervals. The run folders and chain logs are in `docs/video_sweep_runs.tar.gz`
(19 finished runs, 20 MB, sha256 `e6a4563ac50fa4732d35c1649aea8c4f6b3a081ab077c1acdb0f9081955b0856`).

## What ran

| model | GPU | smoke run twice | full `video_sweep` (64 scenes) | `media_patch` |
|---|---|---|---|---|
| Qwen3.5-0.8B | L40S | identical (5,700 rows) | done, 66 min | done, 1.4 min |
| Gemma-4-E2B | L40S | identical (6,570 rows) | done, 3.5 h | done, 3.9 min |
| Qwen3.5-4B | L40S | identical | done, 3.7 h | done, 3.9 min |
| Gemma-4-E4B | L40S | identical (7,380 rows) | done, 5.8 h | done, 8.0 min |
| Qwen3.5-27B | H200 | identical (9,750 rows) | **not run** (budget: about 6.5 h at $5.29/hr) | done, 7.0 min |
| Gemma-4-31B | — | not run | not run | not run |

The smoke runs (5 scenes) give counts only, no verdicts. The first Qwen3.5-0.8B smoke pair differed in 1 of 5,700 answers,
because each engine re-measured the free GPU memory and the batch sizes moved; after PR #4's `e4d3f85` (one token budget per
run) every pair is identical. That first pair is kept in the archive under `runs/attempt_budget/`.

Mean GPU utilization over the full `video_sweep` runs: 82% (0.8B), 93% (E2B), 93% (4B), 96% (E4B); median 100% on all.

## video_sweep: every verdict is untestable

The naming question (`say`) fails its 0.90 clean-accuracy gate on every input of every model, so by PR #4's rule every
patching and steering verdict is "not perceived, untestable", and every prediction (P1–P5, and #3's 1–8) is untestable. The
positive control (the every-layer donor patch makes the name follow the donor on ≥ 0.9) fails too, on every input.

Clean accuracy (a `*` marks a failed gate):

| model | input | say | animal | letter | set | ride |
|---|---|---|---|---|---|---|
| 0.8B | moving | 0.83* | 1.00 | 0.47* | 0.75* | 0.61* |
| | frozen | 0.78* | 0.94 | 0.38* | 0.70* | 0.66* |
| | still | 0.83* | 0.00* | 0.17* | 0.00* | 0.00* |
| E2B | moving | 0.70* | 0.98 | 0.36* | 0.88 | 0.72* |
| | frozen | 0.62* | 1.00 | 0.34* | 0.91 | 0.72* |
| | still | 0.77* | 0.98 | 0.55* | 0.92 | 0.75* |
| 4B | moving | 0.89* | 1.00 | 0.88 | 0.92 | 0.77* |
| | frozen | 0.86* | 1.00 | 0.86 | 0.94 | 0.88 |
| | still | 0.83* | 1.00 | 0.89 | 0.94 | 0.91 |
| E4B | moving | 0.58* | 1.00 | 0.72* | 0.98 | 0.81* |
| | frozen | 0.58* | 1.00 | 0.64* | 0.92 | 0.80* |
| | still | 0.70* | 1.00 | 0.77* | 0.97 | 0.89 |

Every-layer donor patch, name follows donor (moving / frozen / still): 0.8B 0.84 / 0.84 / 0.84; E2B 0.77 / 0.67 / 0.80;
4B 0.88 / 0.89 / 0.89; E4B 0.72 / 0.64 / 0.81. Self-patching gives the clean answer on 0.97–1.00 of rows everywhere, so the
machinery works; what fails is the models' naming of these scenes.

**Horizons, reported though the gates fail** (the last window start at which a question follows the donor on ≥ 0.5 of
scenes; moving input, the frozen one is the same unless noted):

| model | windows | say | animal | letter | set | ride |
|---|---|---|---|---|---|---|
| 0.8B (24 layers) | −1…23 | 15 | 3 | none | 3 | 3 |
| E2B (35) | −1…31 | 11 | 7 | none (still: 11) | 7 | 7 |
| 4B (32) | −1…31 | 19 | 7 | 23 (still: 19) | 7 | 7 |
| E4B (42) | −1…39 | 19 | 15 | 19 | 11 (still: 15) | 15 |

On every model the name stays movable at a later window than the animal, set and ride answers, but with the gates failed
no prediction can be scored on it.

**The 0.8B still input's zeros are a prompt problem, not a finding.** On the still picture, Qwen3.5-0.8B answers the animal
question with "The object in image…" on all 64 scenes, not a digit, so it scores 0; set and ride show the same zeros. The
other three models answer the same still prompt correctly (animal 0.98–1.00), so it is the 0.8B model with that prompt.

## media_patch (VGGSound clips, and the static control)

| model | clip gates say / animal | clip every-layer | L_say | L_use | clip verdict | P1 | P2 | P3 | P4 | static |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | 1.00 / 1.00 | 1.00 | 15 | 3 | too late | kept | kept | kept | untestable | say gate fails (0.89) |
| E2B | 0.98 / 1.00 | 1.00 | 11 | 7 | together | **missed** | kept | kept | untestable | say gate fails (0.81) |
| 4B | 0.98 / 1.00 | 1.00 | 19 | 7 | too late | kept | kept | kept | kept | too late; P1–P4 kept |
| E4B | 1.00 / 1.00 | 1.00 | 19 | 15 | together | **missed** | kept | kept | untestable | positive control failed (0.88) |
| 27B | 1.00 / 1.00 | 1.00 | 47 | 31 | too late | kept | **missed** | kept | untestable | positive control failed (0.81) |

On the clips every gate and control passes on all five models. The Qwen models give "too late" (the answers that use the
name stop following the donor well before the name does); the two Gemma models give "together", which misses P1. Qwen3.5-27B
misses P2. Only Qwen3.5-4B passes the static control, and there it gives the same horizons and verdict as on the clips.

## Cost

L40S about $21 over four pods; H200 about $13 (the 27B smoke pair and its `media_patch`); the network volume about $0.70 a day.
