# Omni lens fits: the timing runs

PR #12's `scripts/fit_lenses.sh` with `TIMING_ONLY=1` (`89e0f47`): a 10-prompt `jlens.fit` per Qwen omni model, to measure the
seconds per prompt before any full fit. Run on 24 November 2026 (12:20 to 12:37 AM PST) on one H200 (141 GB) on the US-NC-1
volume, from its own worktree, the three models one after another. The models, the corpus (WikiText-103 raw at `b08601e0`,
sha256 checked), the prompt rule and the settings (`max_seq_len` 128, `dim_batch` 8) are PR #12's. Numbers are copied from each
run's `meta.json`. Run folders (`meta.json`, `report.csv`, `run.log`, without the 10-prompt `lens.pt`) and the chain logs:
`docs/omni_lens_timing_30b.tar.gz` (sha256 `10c7f5fc203eea28e4342109a9c031bd1b765005950762669173d5438cc12370`) and
`docs/omni_lens_timing_3b_7b.tar.gz` (sha256 `4cc9a030a117df8dece361922d8f482ee5f1773ca277ce29e847271b3d82be7b`).

## Seconds per prompt and the projected full fits

| model | layers / width | source layers | seconds per prompt | 100 prompts | 250 prompts | 1,000 prompts |
|---|---|---|---|---|---|---|
| Qwen2.5-Omni-3B | 36 / 2048 | 35 | 7.6 | 0.21 h | 0.53 h | 2.11 h |
| Qwen2.5-Omni-7B | 28 / 3584 | 27 | 17.8 | 0.49 h | 1.23 h | 4.93 h |
| Qwen3-Omni-30B-A3B-Instruct | 48 / 2048 | 47 | 20.0 | 0.55 h | 1.39 h | 5.55 h |

All on one H200. The three 250-prompt fits come to about 3.2 hours, about $17 at $5.29/hr, plus model loading and the report
(about 15 minutes per model).

## The 10-prompt lenses' report (no pass/fail)

`report.csv`: on the last 28 positions of 32 WikiText-103 validation paragraphs, the share of positions where the readout's top
token is the model's own next-token prediction, at the deepest fitted layer:

| model | layer | J-lens | logit lens |
|---|---|---|---|
| Qwen2.5-Omni-3B | 34 of 36 | 0.627 | 0.635 |
| Qwen2.5-Omni-7B | 26 of 28 | 0.628 | 0.582 |
| Qwen3-Omni-30B-A3B-Instruct | 46 of 48 | 0.637 | 0.600 |

These lenses are fitted on 10 prompts only and are not kept; the full fits decide, and `scripts/check_lens.py`'s rule (written
before it runs) decides whether a lens is used. For scale only, and on another test (check_lens's prompts and positions): the
Gemma-4-12B lens that failed check_lens agreed on 0.139 at its deepest layer, Gemma-4-E4B's Neuronpedia lens on 0.486.

## Notes

- Each thinker loads with transformers' "UNEXPECTED" keys note: the checkpoint's speech-output weights are not loaded, as PR #12
  intends.
- `omni_timing.sh` (the pod chain) piped `fit_lenses.sh` through `tee runs/...`, which failed on the new worktree's missing
  `runs/` for the 30B run; its output is in the chain log, and the run itself was not affected.

## Run

About 21 minutes of H200 time (about $1.85), and a CPU pod for the downloads beforehand (about $0.10). The pod stopped itself.
