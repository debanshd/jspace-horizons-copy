# Omni lenses at 500 prompts: the deciding lenses and their verdicts

PR #12's extension (`EXTEND_TO=500 scripts/fit_lenses.sh`, at `e03e8b5`): each 250-prompt omni lens (PR #13's fits, prompts
0–249) gets prompts 250–499 fitted apart and merged into the 500-prompt lens, which `config.LENS_CHECKS` makes the deciding one
since `e03e8b5` (changed after check 3's result, approved, recorded there). Run 24 November 2026, 8:42 PM to 12:21 AM PST, on one
H200 on the US-NC-1 volume, in the omni worktree (about $15). Check 1 (Qwen3.5-2B's reproduction, `docs/lens_checks_results.md`)
had passed. Numbers are copied from each `runs/lens_validation/<model>/summary.json` and `runs/lens_halves/<model>/compare.json`.

**Rule** (`scripts/validate_lens.py`): a lens is used if check 1 passed and check 2 passes; that is, its two halves (prompts
0–249 and 250–499, fitted apart) agree at cosine ≥ 0.99 at every read layer. Check 3 (concept readout over the logit lens) is
reported beside it and decides nothing.

| model | added prompts' fit | check 2: lowest cosine of the halves | check 3 (caveat): J-lens / logit lens pass@10, margin | verdict |
|---|---|---|---|---|
| Qwen2.5-Omni-3B | 1.0 h | 0.9949 (layer 10) | 0.226 / 0.165, +0.061 | **passes** |
| Qwen2.5-Omni-7B | 2.4 h | 0.9953 (layer 8) | 0.280 / 0.215, +0.066 | **passes** |
| Qwen3-Omni-30B-A3B-Instruct | 2.7 h | **0.9887** (layer 14; 15: 0.9900, 16: 0.9908) | 0.327 / 0.278, +0.048 | **fails** |

The 30B's lens has not stopped changing at 500 prompts by this rule: a miss, recorded as such. Check 3 is void for the family
(its bar, Neuronpedia's 4B at −0.033, is below 0), but all three omni lenses read the concepts better than their own logit lens.

## The lenses, on Hugging Face

In the private repo [`debanshd/jspace-omni-lenses`](https://huggingface.co/debanshd/jspace-omni-lenses) at commit [`a473c0784d05057a15f278b703a9cfae6499880c`](https://huggingface.co/debanshd/jspace-omni-lenses/tree/a473c0784d05057a15f278b703a9cfae6499880c), under `<model>/n500/` with
`meta.json` and `report.csv`; the sha256 Hugging Face stores for each `lens.pt` matches its `meta.json`:

| model | file | sha256 |
|---|---|---|
| Qwen2.5-Omni-3B | [`Qwen2.5-Omni-3B/n500/lens.pt`](https://huggingface.co/debanshd/jspace-omni-lenses/blob/a473c0784d05057a15f278b703a9cfae6499880c/Qwen2.5-Omni-3B/n500/lens.pt) | `a7535af6f48b566e4da89fd39a54c3b418fe3ce5d001efcf34c0983bec7c141c` |
| Qwen2.5-Omni-7B | [`Qwen2.5-Omni-7B/n500/lens.pt`](https://huggingface.co/debanshd/jspace-omni-lenses/blob/a473c0784d05057a15f278b703a9cfae6499880c/Qwen2.5-Omni-7B/n500/lens.pt) | `e2e88d60d9567a3a8225e46725938453ff7b1fc3bf899b6d710af1bf41596ba5` |
| Qwen3-Omni-30B-A3B-Instruct (fails check 2) | [`Qwen3-Omni-30B-A3B-Instruct/n500/lens.pt`](https://huggingface.co/debanshd/jspace-omni-lenses/blob/a473c0784d05057a15f278b703a9cfae6499880c/Qwen3-Omni-30B-A3B-Instruct/n500/lens.pt) | `257e34cc4209b9ab4e006fd41fc36d9bec7f2ba8686cbcea86ee3406d93c179e` |

The 30B's lens is uploaded for the record; by the verdict it is not used.

`docs/omni_lens_n500.tar.gz` (sha256 `49857bafb4e571d4add4f29a9a5cb07a1eb35748efc365e95bf015b5f15b5b9b`): every `meta.json`, `report.csv` and `run.log` of the extension and the
merges, `runs/lens_halves`, `runs/lens_readout`, `runs/lens_validation`, the chain log; no `lens.pt`.
