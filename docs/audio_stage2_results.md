# Audio, stage 2: the rule-wording check, the spoken/frozen/typed pilot, and the 12B's lens

PR #10 at `9224a34`, run on 24 November 2026 (8:41 to 8:59 PM PST) on one L40S on the US-NC-1 volume, from its own worktree
(the video checkout untouched), with `SWEEP=audio_sweep_static`, `KEEP_GOING=1` and `FULL=0` (stop after the pilot). Models:
Gemma-4-E2B, Gemma-4-E4B (their stage-1 `audio_clean` runs reused, `docs/audio_stage1_results.md`) and Gemma-4-12B. The
experiments' rules are PR #10's, written before the run. Numbers are copied from each run's `summary.json`; the run folders
and the chain log are in `docs/audio_stage2.tar.gz` (sha256 `220605f470e3371e2b482efe2042eca9275c7e1b77f30b0129d7633bdf148b7c`).

**No full sweep ran.** Gemma-4-E2B and E4B stopped at the pilot (the frozen word, below), Gemma-4-12B at its lens check.

## `audio_rules`: why the audio-first rule answers fail

Each rule in three wordings: A the dataset's, B the digits swapped, D the 3 named first. Share correct over 128 answers:

| model | order | animal A / B / D | set A / B / D | ride A / B / D | diagnosis |
|---|---|---|---|---|---|
| Gemma-4-E2B | question first | 1.00 / 1.00 / 1.00 | 0.99 / 0.99 / 1.00 | 0.95 / 0.89 / 0.56 | mixed |
| | audio first | 0.73 / 0.73 / 0.50 | 0.78 / 0.87 / 0.52 | 0.91 / 0.88 / 0.50 | **first named** |
| | typed | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.88 | too few errors |
| Gemma-4-E4B | question first | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | too few errors |
| | audio first | 0.73 / 0.66 / 1.00 | 0.78 / 0.73 / 0.88 | 0.55 / 0.55 / 0.55 | mixed |
| | typed | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | too few errors |

Wording A repeats `audio_clean`'s prompts, and its answers match that run's on every row (1.00 in each order).

- **Gemma-4-E2B, audio first: a first-option bias.** When the 3 is named first (D), 100% of its wrong answers are 3 (98.5%
  when the digits are swapped, B): it answers whichever digit the rule names first. It fails the animal gate in every wording.
- **Gemma-4-E4B, audio first: no single bias** ("mixed"). Its wrong answers are all 3 under B but mostly 7 under D (21% are
  3), and it passes the animal gate under D (1.00) but not under A or B.
- With the question first or the word typed, both pass the animal gate in every wording; Gemma-4-E2B's ride answers drop
  under wording D even there (0.56 question first).

## `audio_sweep_static` pilot (5 scenes, patching only, twice)

| model | input | clean say | clean animal | self-patch | every-layer donor patch | donors named |
|---|---|---|---|---|---|---|
| Gemma-4-E2B | spoken (`q_audio`) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | frozen (`q_frozen`) | **0.00** | 0.60 | 1.00 | **0.00** | **0.00** |
| | typed (`q_typed`) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemma-4-E4B | spoken | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | frozen | **0.40** | 0.40 | 1.00 | **0.20** | **0.20** |
| | typed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The two pilot runs are identical (Gemma-4-E2B 2,325 rows, E4B 2,775). The frozen word (every 40-ms block of the spoken span
replaced by its middle block) is not recognized: the models neither name it nor follow a frozen donor. `audio_sweep_static`'s
rule, written before the run, expected this as possible and makes such an input "not perceived, untestable", a result. The
launcher's pilot gate at `9224a34` instead required every input's controls to pass, so it stopped both models before the full
sweep. PR #10's `2d1243d` (pushed during this run) makes the gate follow the rule: the spoken input must pass every control,
an unperceived counterpart input is untestable, and the chain goes on. With it, both models' pilots pass.

## `check_lens`: Gemma-4-12B's lens from `hlampes/jacobian-lens`

Agreement of each lens's top token with the model's own next-token prediction, by read layer (rule written before the run:
deepest agreement near the reference model's, and above the logit lens from half depth):

| model, lens | deepest read layer | J-lens there | logit lens there | above the logit lens from half depth | passes |
|---|---|---|---|---|---|
| Gemma-4-E4B, Neuronpedia's (the reference) | 39 of 42 | 0.486 | 0.370 | yes | yes |
| Gemma-4-12B, `hlampes/jacobian-lens` (250 WikiText prompts) | 46 of 48 | 0.139 | 0.005 | **no** (0.000 vs 0.036 at 34, 0.060 vs 0.079 at 38) | **no** |

So Gemma-4-12B stopped before its audio runs; its lens is not good enough to use.

## Run

About 20 minutes of L40S time (about $0.40), the pod stopped itself when the chain ended; the Gemma-4-12B download (about
24 GB and its lens, all checked) was done beforehand on a CPU pod.
