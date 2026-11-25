# check_lens on the Qwen lenses: the four runs of 24 November

PR #14's `scripts/check_lens.py` runs on 24 November 2026 (all on one H200 on the US-NC-1 volume, in the audio worktree), before
PR #12's lens validation (`docs/lens_checks_results.md`) replaced `check_lens` as the test of a fitted lens. The rule
(written before any run): at the deepest read layer the J-lens agrees with the model's next token on at least E4B's Neuronpedia
lens's share minus 0.10, and it beats the plain logit lens at every read layer from half depth. Numbers are copied from each
run's `summary.json`; every run folder is in `docs/check_lens_runs.tar.gz` (sha256 `68a0d734e23aaf8269c52181bf6ad0758fa8d2771170229cfb4eaf81dabf4b2d`), one folder per run.

## 1. First pass (`50d7af5`, 2:47 to 2:55 PM PST): loading test on the 250-prompt omni lenses

| model | loaded | deepest | beats the logit lens | passes (decides nothing) |
|---|---|---|---|---|
| Gemma-4-E4B (reference) | yes | 0.478 | yes | |
| Qwen2.5-Omni-3B | yes | 0.562 | no | no |
| Qwen2.5-Omni-7B | yes | 0.590 | no | no |
| Qwen3-Omni-30B-A3B | **no**: its tokenizer has no chat template | | | |

`audio_clean` also ran on the 3B and 7B, and is **void**: the checkpoints' `generation_config.json` names no end token, so every
answer ran to the token limit and was scored wrong although most were right ("Airplane", "3", "A"). The folders are kept as
`runs/void_noeos_audio_clean`; `79a3dd2` fixed the end token and the chat template.

## 2. Lens checks (`7e190bb`, 4:04 to 4:11 PM PST)

- **Qwen3.5-9B's J-lens** (`bcywinski/jacobian-lens-qwen3.5-9b`, for the mid-size video runs): **fails**, deepest 0.353 against
  E4B's 0.486 (more than 0.10 below), and it does not beat the logit lens. A miss, recorded as such.
- **Qwen3-Omni-30B-A3B** loaded but stopped at Session's picture check: it picked the right letters (ACD) with only 0.32 of its
  first-token probability on them (the check asks 0.5).

## 3. Calibration on Neuronpedia's own Qwen lenses (`88bd961`, 4:40 to 4:50 PM PST)

| Neuronpedia lens | deepest | beats the logit lens | passes |
|---|---|---|---|
| Qwen3.5-2B (J-lens) | 0.486 | yes | **yes** |
| Qwen3.5-4B (J-lens) | 0.312 | no | no |
| Qwen3.5-9B (J++) | 0.177 | no | no |
| Qwen3.5-27B (J-lens) | 0.264 | no | no |

By the consequence written before this run, the rule is not calibrated for Qwen. The Qwen3-Omni diagnosis (`diagnose_preflight`):
with no answer prefix (Session's choice from text-only questions) 0.24 to 0.40 of the first token falls on the letters, with "("
0.59 to 0.74; its greedy answers name the letter in words ("The image shows a capital letter 'A'…").

A first replacement rule (`38d96f9`, the weakest Neuronpedia agreement as the bar, 0.00 at most layers) was withdrawn before it
judged anything.

## 4. The rank-based Qwen rule's calibration (`7c1cff9`, 6:09 to 6:18 PM PST)

The median log10 rank of the model's top token in each readout, for the four Neuronpedia lenses: the bar, set from them, does
not reject Qwen3.5-9B's own logit lens, so the rule is **void** by its own condition and judges nothing. In the same run
Qwen3-Omni-30B-A3B loaded with its answer prefix chosen on card B ("(", letter mass 0.52) and read its 250-prompt lens: deepest
0.579, beats the logit lens, passes the first rule (a lens already seen: decides nothing).

About $4.50 of H200 time for the four runs.
