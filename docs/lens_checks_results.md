# Lens validation: the reproduction checks and the concept-readout calibration

PR #12's lens checks (`config.LENS_CHECKS`, `config.LENS_FAMILIES`, written before they ran; `docs/lens_validation_run.md`),
run on 24 November 2026. Numbers are copied from each run's `compare.json` and `summary.json`.

- **Check 1, reproduce:** this project's fitting code refits a lens Neuronpedia fitted, with as many prompts, and the two maps
  agree: cosine ≥ 0.99 at every layer the experiments read (from 0.3 of the depth).
- **Check 3, reads concepts:** on the J-lens paper's lens-quality prompts (five sets), pass@10 minus the model's own logit lens's.
  The bar is the lowest margin among Neuronpedia's lenses of the family, and the check is void unless every one of them is above 0.
  Since `e03e8b5` (a change after the Qwen result, approved and recorded as such) it is reported as a caveat and decides nothing.

## Qwen (check 1 passes; check 3 void)

Run on an RTX PRO 6000 in EUR-IS-2, with no volume, at PR #12's `c87f853`, 7:05 to 8:06 PM PST (about $3). A first try on an
H200 stopped at the first backward pass: the library behind Qwen3.5's linear-attention layers refuses it on Hopper GPUs with the
Triton `uv.lock` pins (wrong results there), so Qwen3.5 lenses are fitted on non-Hopper GPUs.

**Check 1:** Qwen3.5-2B refitted with 283 prompts (10.4 s per prompt; lens sha256 `eca4bbeb…`) against Neuronpedia's
`qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt` at `e166deb5`: **passes**, lowest cosine 0.9920 (layer 7),
rising to 1.000 at layer 21; with the identity part removed, 0.993 to 0.999.

**Check 3's calibration:**

| Neuronpedia lens | J-lens pass@10 | logit lens pass@10 | margin |
|---|---|---|---|
| Qwen3.5-2B | 0.337 | 0.235 | +0.102 |
| Qwen3.5-4B | 0.324 | 0.358 | −0.033 |
| Qwen3.5-27B | 0.382 | 0.406 | −0.025 |

The bar is −0.033, below 0: **void**. The 4B's and 27B's lenses, which the video results read, do not read these concepts better
than their logit lens.

## Gemma (check 1 fails; check 3 void)

Run on an H100 80GB in AP-IN-1, with no volume, at PR #14's `8853711` (which carries PR #12's Gemma checks), 9:56 to 11:37 PM PST
(about $5). The H100, because the account's plan allows one H200 at a time, which the omni extension held.

**Check 1:** the base Gemma-4-E2B (`63db66a3`, the model Neuronpedia fitted) refitted with 455 prompts (10.6 s per prompt; lens
sha256 `5d22b5c9…`) against Neuronpedia's `gemma-4-e2b/jlens/Salesforce-wikitext/gemma-4-E2B_jacobian_lens.pt` at `e166deb5`:
**fails**, lowest cosine 0.9436, at layer 14.

| layer | 10 | 11 | 12 | 13 | **14** | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24–33 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cosine | 0.991 | 0.990 | 0.991 | 0.992 | **0.944** | 0.981 | 0.988 | 0.989 | 0.990 | 0.989 | 0.989 | 0.990 | 0.984 | 0.991 | 0.996–1.000 |

With the identity part removed the cosine is 0.991–0.993 at layers 10–12 and 1.000 from layer 13 on, a few values slightly above
1 (up to 1.001), which a cosine cannot be. So the comparison's numbers at those layers need understanding before the failure is
read as the recipe's; the rule stands as written, and **the Gemma-4-26B-A4B and Gemma-4-12B fits did not run**.

**Check 3's calibration:**

| Neuronpedia lens | J-lens pass@10 | logit lens pass@10 | margin |
|---|---|---|---|
| Gemma-4-E2B-it | 0.467 | 0.278 | +0.190 |
| Gemma-4-E4B-it | 0.378 | 0.306 | +0.072 |
| Gemma-4-31B-it | 0.365 | 0.519 | −0.155 |

The bar is −0.155: **void**.

## Files

- `docs/lens_check_qwen.tar.gz` (sha256 `940bb4789c24b470619675f759ddb6caca3165712bfea4dcc537773e874402c8`): `runs/lens_repro`, `runs/lens_readout_np`, the refit's `meta.json`
  and `run.log` (no `lens.pt`), `runs/check_recipe.log` and the chain logs.
- `docs/lens_check_gemma.tar.gz` (sha256 `760d374c2387c5b73ccc93cb4f05b7726f680bba62b1de84e2329454f450910e`): the same for Gemma, with `runs/check_recipe_gemma.log`.
