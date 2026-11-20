# Audio, stage 1: the clean gate on the matched spoken-audio set

PR #10's `audio_clean` (`fe9a7c0`) on Gemma-4-E2B and Gemma-4-E4B, run on 20 November 2026 (12:09 to 12:14 AM PST) on one
L40S on the US-NC-1 volume, from its own worktree, on `saxenan3/jspace-matched-spoken-audio` at `d95fa45e` (every recording
checked against `checksums.json`). The questions, the whole-answer scoring (`src/jspace/strict.py`) and the rule are PR #10's,
written before the run. Numbers are copied from each run's `summary.json`; the run folders and the chain log are in
`docs/audio_stage1.tar.gz` (sha256 `de8223da26c0fccf822df5bd70bb73b8d85637a753d85be0cc0d68ce8b526395`).

**Neither model is ready, so `scripts/audio_sweep.sh` stopped each one before the patch-only pilot: no patching or steering
ran.** Its rule needs, on the gated scenes and in both prompt orders, receivers and donors to pass say >= 0.90 and
animal >= 0.85; both models fail the animal gate when the audio comes before the question.

## Recognition of the 15 recordings (exact name)

| model | question, then audio (`q_audio`) | audio, then question (`audio_q`) | typed transcript | scenes gated (both orders) |
|---|---|---|---|---|
| Gemma-4-E2B | 14 of 15 (missed "bus": "Bass") | 14 of 15 (the same miss) | 15 of 15 | 57 of 64 |
| Gemma-4-E4B | 15 of 15 | 15 of 15 | 15 of 15 | 64 of 64 |

Controls: 3 s of silence and 3 s of white noise named no object of the set (answers "Book", "book", "table", "thought").

## Clean answers on the gated scenes (receiver; donor in brackets), with their intervals

| model | order | say | animal (gate 0.85) | letter | set | ride |
|---|---|---|---|---|---|---|
| Gemma-4-E2B | `q_audio` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.98 (1.00) | 0.93 (0.95) |
| | `audio_q` | 1.00 (1.00) | **0.75** [0.48, 0.95] (0.75) | 1.00 (1.00) | 0.79 (0.82) | 0.91 (0.88) |
| Gemma-4-E4B | `q_audio` | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| | `audio_q` | 1.00 (1.00) | **0.73** [0.44, 0.93] (0.72) | 1.00 (1.00) | 0.77 (0.80) | 0.58 (0.52) |

The typed transcript scores 1.00 on every question for both models.

## What the failing answers are (read from `audio_clean.csv`)

With the audio first, the rule answers lean to "7", the first answer each rule names ("if ..., answer 7; otherwise answer 3"),
while the name itself is still recognized exactly. Counts over the 128 answers per question (64 scenes, receiver and donor; the
truth is 64 "7" and 64 "3" for animal):

| model | question | `q_audio` "7" / "3" | `audio_q` "7" / "3" |
|---|---|---|---|
| Gemma-4-E2B | animal | 64 / 64 | 99 / 29 |
| Gemma-4-E2B | set | 65 / 63 | 92 / 36 |
| Gemma-4-E4B | animal | 64 / 64 | 99 / 29 |
| Gemma-4-E4B | ride | 64 / 64 | 122 / 6 |

So the failure is in applying a rule to a heard word when the word comes first, not in hearing it.

The intervals here are over the 15 recordings, not independent scenes: the 64 scenes reuse them.

## Run

About 10 minutes of L40S time in all (E2B 71 s and E4B 84 s for `audio_clean`, plus building the worktree's environment and
loading the models), about $0.20. The pod stopped itself when the chain ended.
