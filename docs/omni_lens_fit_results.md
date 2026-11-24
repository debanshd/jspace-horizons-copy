# Omni lens fits: the full fits

PR #12's `scripts/fit_lenses.sh` (`89e0f47`), N = 250 prompts per model, run on 24 November 2026 (1:19 to 4:31 AM PST) on one
H200 on the US-NC-1 volume, from its own worktree, the three Qwen omni models one after another. The models, the corpus
(WikiText-103 raw at `b08601e0`, sha256 checked), the prompt rule and the settings (`max_seq_len` 128, `dim_batch` 8) are
PR #12's. Numbers are copied from each run's `meta.json` and `report.csv`.

## The lenses, on Hugging Face

In the private model repo `debanshd/jspace-omni-lenses` at commit `4648419f6346d3cbb356ed7b16b8a8b1fa2cbe75`, each under
`<model>/` with its `meta.json` and `report.csv`. The sha256 Hugging Face stores for each `lens.pt` matches the one `meta.json`
records:

| model | file | size | sha256 |
|---|---|---|---|
| Qwen2.5-Omni-3B | `Qwen2.5-Omni-3B/lens.pt` | 294 MB | `8a10a569f4fe3614bf4575280079b5a2ddad7b39c173bec59f25da237992cde7` |
| Qwen2.5-Omni-7B | `Qwen2.5-Omni-7B/lens.pt` | 694 MB | `d1696e876294f9e1d3510339905d64613d49125d9b74898e435a0583ef94e859` |
| Qwen3-Omni-30B-A3B-Instruct | `Qwen3-Omni-30B-A3B-Instruct/lens.pt` | 394 MB | `2c2fa06026a42307282e79d2ad120879fe01826287ffd15018c578ad36d50bcb` |

The experiments pin the repo, that commit and each sha256 in `config.py` (the next PR, with `Session` loading a thinker).
`meta.json`, `report.csv`, each fit's `run.log` and the chain logs, without the lenses: `docs/omni_lens_fit.tar.gz` (sha256
`83a7f04137de24e86d857b6ffbfe9a2331605278f3591b554bdb17ce61d87e8c`).

## The fits

| model | prompts | seconds per prompt | hours |
|---|---|---|---|
| Qwen2.5-Omni-3B | 250 | 7.2 | 0.50 |
| Qwen2.5-Omni-7B | 250 | 17.3 | 1.20 |
| Qwen3-Omni-30B-A3B-Instruct | 250 | 19.6 | 1.36 |

## The report (no pass/fail)

On the last 28 positions of 32 WikiText-103 validation paragraphs: the share of positions where the readout's top token is the
model's own next-token prediction.

| model | deepest fitted layer | J-lens there | logit lens there | J-lens above the logit lens, layers from half depth |
|---|---|---|---|---|
| Qwen2.5-Omni-3B | 34 of 36 | 0.629 | 0.635 | 14 of 17 |
| Qwen2.5-Omni-7B | 26 of 28 | 0.617 | 0.581 | 9 of 13 |
| Qwen3-Omni-30B-A3B-Instruct | 46 of 48 | 0.631 | 0.600 | 15 of 23 |

Whether a lens is used is decided by `scripts/check_lens.py`'s rule, on its own prompts and positions, once the experiments can
load these models; this report is not that check. For scale only, on check_lens's test: Gemma-4-E4B's Neuronpedia lens 0.486 at
its deepest read layer, the failed Gemma-4-12B lens 0.139.

## Run

About 3.5 hours of H200 time (1:19 to 4:51 AM PST, the last 20 minutes holding the pod for the 2.6 GB download), about $18.50.
The pod stopped itself afterwards (a watchdog on the chain; the hold for the download had its own).
