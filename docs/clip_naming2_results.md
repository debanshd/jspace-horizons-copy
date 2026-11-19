# clip_naming2: results on the two Gemma models

PR #6's `clip_naming2` (`2fa2f96`), run on 19 November 2026 (8:26 to 8:31 PM PST) on one L40S, the same GPU type as
`clip_naming`, with the pinned model revisions and the matched set at `e33c6ae8` (all 73 clips checked against
`ATTRIBUTION.tsv`). The question, scoring and rule are PR #6's. Numbers are copied from each run's `summary.json`; the run
folders are in `docs/clip_naming2_runs.tar.gz` (sha256 `05e3555f45a7699abc802c2e329c3fe0e22a95f9426382f72122ed59f61a170a`).

## Share of the 73 clips named correctly (gate 0.90)

| model | moving | frozen | still | reliable clips | gated scenes (of 64) |
|---|---|---|---|---|---|
| Gemma-4-E2B | 0.904 [0.84, 0.96] | 0.918 [0.85, 0.97] | 0.904 [0.84, 0.96] | 62 | 43 |
| Gemma-4-E4B | 0.945 [0.89, 0.99] | 0.945 [0.89, 0.99] | 0.959 [0.90, 1.00] | 69 | 57 |

Both models pass the gate on all three inputs. Against `clip_naming` (`docs/clip_naming_results.md`): E2B 0.904 / 0.836 /
0.932 → 0.904 / 0.918 / 0.904, E4B 0.849 / 0.849 / 0.918 → 0.945 / 0.945 / 0.959.

**How much of this the run itself decided.** The new scoring was written after reading `clip_naming`'s answers, and applied
to those answers it already gives E4B 69 / 69 / 70 of 73 and E2B 67 / 61 / 68 (checked here). E4B's run gives 69 / 69 / 70:
its pass was set by the scoring, and the new question changed none of its counts. E2B's frozen input, 61 under the new
scoring alone, is 67 here: that gain is the question's.

## The misses (read from `clip_naming2.csv`)

- **Gemma-4-E2B** (20): generic words remain despite the question ("food" ×2 and "dish" for pizza, "food" and "coffee" for a
  cup, "vehicle" ×2, "mammal" ×2, "railway", "Garden ornament"); "dog" (or "Canine", "Canis lupus familiaris") for a horse
  on 6 answers and for a cat on 2; "truck" for a bus.
- **Gemma-4-E4B** (11): "sheep" for a horse ×3, "subway car" for a train ×3, "off-road vehicle", "SUV" and "truck" for a
  car, "raccoon" and "bear" for a cat.

Answers counted correct through the new head-noun scoring include "race car", "sports car", "coffee cup", "office chair",
"electric bicycle", "mountain bike" and "wall clock"; none of the loophole cases ("toy car", "stuffed horse") occur.
