# clip_naming: results on the four smaller models

PR #6's experiment, run on 19 November 2026 (6:52 to 7:12 PM PST) with `scripts/clip_naming.sh` on one L40S, at PR #6's
code `4efb143`, the pinned model revisions, and the matched video set at `e33c6ae8` (all 73 clips checked against
`ATTRIBUTION.tsv`). The question, the scoring and the rule are PR #6's, written before this run. Every number is copied from
a run's `summary.json`; intervals are 95% bootstrap intervals over clips. The run folders are in
`docs/clip_naming_runs.tar.gz` (sha256 `f082d203474b2aead2c43e52bf0f1ca86e562248aa82f6a251d714aa227e45a7`).

## Share of the 73 clips named correctly (gate 0.90; `*` fails)

| model | moving | frozen | still | reliable clips | gated scenes (of 64) | time |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.863 [0.78, 0.93]* | 0.781 [0.68, 0.88]* | 0.877 [0.79, 0.95]* | 52 | 31 | 3.6 min |
| Gemma-4-E2B | 0.904 [0.84, 0.97] | 0.836 [0.74, 0.92]* | 0.932 [0.86, 0.99] | 57 | 38 | 3.0 min |
| Qwen3.5-4B | **0.973** [0.93, 1.00] | **0.973** [0.93, 1.00] | **0.986** [0.96, 1.00] | 70 | 57 | 5.2 min |
| Gemma-4-E4B | 0.849 [0.77, 0.93]* | 0.849 [0.77, 0.93]* | 0.918 [0.85, 0.97] | 60 | 41 | 3.1 min |

Only Qwen3.5-4B passes the gate on all three inputs. The scenes whose receiver and donor are both reliable for all four
models: 13 of 64 (3, 11, 14, 15, 25, 27, 32, 36, 38, 42, 46, 52, 63).

Against `video_sweep` (PR #5, the canvas and its noun question, moving input): 0.8B 0.83 → 0.86, E2B 0.70 → 0.90,
4B 0.89 → 0.97, E4B 0.58 → 0.85. The two runs differ in both the display and the question (and `video_sweep` counts 64
scenes, this 73 clips), so the change is their joint effect.

## What the misses are (read from `clip_naming.csv`)

- **Qwen3.5-4B** (5): a car called "truck" on all three inputs, a cat called "dog" twice.
- **Qwen3.5-0.8B** (35): lists and repetitions ("Horse, cow, horse, hor…", "laptop;computer;…"), generic words ("animal",
  "Vehicle", "food"), "automobile" and "aircraft", and real confusions ("dog" for a cat, "bowl" for a cup).
- **Gemma-4-E2B** (24): generic words ("animal" ×5, "food" ×4 for pizza, "railway", "Furniture"), "dog" for horses ×4, and
  "person" for a bicycle and a cup.
- **Gemma-4-E4B** (28): mostly more specific kinds that `EXTRA_NAMES` does not list: "sports car" ×3, "race car" ×4,
  "coffee cup" ×3, "office chair" ×2, "subway car" ×2, "electric bicycle", "military aircraft"; also "sheep" for a horse ×3,
  "duck" and "chicken" for a bird, "raccoon" and "bear" for a cat.

The scoring is PR #6's and is not changed here; E4B's specific kinds stay wrong under it.

**Qwen3.5-0.8B's still input** names 64 of 73 clips on the GPU, against 66 in PR #6's CPU pilot (plain `transformers` on a
CPU, a different numeric path); the pilot was exploratory and is not part of the rule.

## The prefetch change (`8e104ce`), checked

On the run above the GPU waited between clips while one CPU core decoded and encoded the next clip. `8e104ce` prepares the
next clip on a worker thread while the GPU answers the current one; each forward pass is unchanged. Qwen3.5-0.8B rerun with it
(`runs/clip_naming_prefetch`, in `docs/clip_naming_prefetch_check.tar.gz`, sha256
`e9f2f9305dc84a6d19bd69f52086664293d3db046c4230b5bbfc383eddce645b`):

- `compare_media.py runs/clip_naming runs/clip_naming_prefetch --file clip_naming.csv`: identical (219 rows).
- 130 s instead of 217 s.
- GPU utilization: mean 33%, median 23% (nvidia-smi every 2 s). A 0.8B model answering one prompt at a time cannot fill an
  L40S even with no wait on the CPU; the gain should be larger on the 27B and 31B models, where each forward is longer.
