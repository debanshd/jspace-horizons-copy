# clip_naming3: results on the four smaller models

PR #6's `clip_naming3` (`c56c8b8`), run on 19 November 2026 (10:37 to 10:53 PM PST) on one L40S, with the pinned model
revisions and the matched set at `e33c6ae8` (all 73 clips checked against `ATTRIBUTION.tsv`). The question, scoring and rule
are PR #6's. Numbers are copied from each run's `summary.json`; the run folders are in `docs/clip_naming3_runs.tar.gz`
(sha256 `e53c8477242b963c7cd07bed8fc23ccfd02c08b9ebfac78f971d83a890ee1215`).

## Share of the 73 clips named correctly (gate 0.90) and the scene pairs left

| model | moving | frozen | still | pairs, moving (the schedule) | pairs, all three inputs | PR #6's floor |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.822* | 0.781* | 0.562* | **42** | 18 | 52 |
| Gemma-4-E2B | 0.904 | 0.918 | 0.904 | 50 | 43 | 50 |
| Qwen3.5-4B | 0.973 | 0.973 | 0.959 | 59 | 56 | 59 |
| Gemma-4-E4B | 0.945 | 0.945 | 0.959 | 57 | 57 | 57 |

`*` fails the gate. No answer on any model hedged, so the follow-up question was never asked.

**The Gemma runs repeat `clip_naming2`.** Same question, same code path, same GPU type: all 219 answers of each Gemma model are
identical to its `clip_naming2` run, and its counts are that run's re-scored.

**Qwen3.5-0.8B falls below the floor because of the question.** The floor (52) was computed from its `clip_naming` answers,
given to the other question. Asked `clip_naming2`'s question, which says "not a general word such as animal, food, vehicle or
furniture", it answers with exactly those words far more often: on the still picture 31 of 73 answers are one of the four
("animal" 21, "food" 4, "vehicle" 3, "furniture" 3), against 3 under `clip_naming`'s question; 16 against 4 on the frozen
input and 12 against 2 on the moving one. The small model takes up the words the question lists. Qwen3.5-4B gives none of the
four under either question.

Run times 2 to 4 minutes per model; mean GPU utilization 31 to 72%.
