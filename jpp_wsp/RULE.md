# Weight-space programming: the OLMo 3 7B 50% J++ lens, written from the 100% lens

Written 26 Nov 2026, before the edit is computed and before anything is scored. Nothing below changes after this commit; a later
change is a new experiment with its own name.

## Question

Can a J++ lens for the 50% checkpoint of OLMo 3 7B be written directly from the 100% checkpoint's J++ lens, using only the two
checkpoints' weights: no prompts, no forward or backward passes, no gradient descent, no fitting loop? This follows the
definition of weight-space programming (an LLM reads and writes network weights without training): Claude, working as a coding
agent, chooses the edit by reasoning about the weights, and a script applies it.

## Inputs (all that the edit may read)

- the source lens: `debanshd/jspace-omni-jpp-runs`, `olmo-3-7b_n64/olmo7b_jpp_n64.pt`
  (sha256 `6b853a64ab98384058949f3349765bddc39cc05d4bb0b4283174444e77378add`), 31 matrices J100_l of 4096 x 4096, source layers 0-30;
- the weights of `allenai/Olmo-3-1025-7B` at `a81bae4` (main, 100%) and at `6bfdeb0` (`stage1-step725000`, 50%).

## What the edit may not read

The target, `olmo-3-7b_n64/olmo7b50pct_jpp_n64.pt`, and everything else about the 50% fit: `fits/olmo-3-7b-50pct/`,
`out_fits/olmo-3-7b-50pct/`, `olmo7b_jpp_out.tgz`, `fit_lenses.log`, `run_all.log`. The only things already seen about it are the
aggregate scores in `out_fits/summary.md` (filtered recall@10 39.5 for the 50% J++ lens, 28.1 for the logit lens). These files are
read only by the scoring step, which runs after the edited lens is frozen and its sha256 is committed here.

## The edit: basis transport

The two checkpoints come from one training run, so their residual streams should share most of their coordinates, up to a slow
drift. Model that drift as a linear map per layer: h50_l = A_l h100_l. A lens maps a layer-l residual into the final residual
(before the final norm), so if J100_l is right for the 100% model, the 50% model's lens is

    J50_l = A_L J100_l A_l^{-1}

with A_L the map at the final residual. Both maps are estimated from weights alone:

- **Input side, B_l = A_l^{-1}.** The residual after block l is read by block l+1 through q_proj, k_proj, v_proj, gate_proj and
  up_proj (OLMo 3 has no pre-norm: these act on the raw residual). If the 50% block reads h50 as the 100% block reads h100, then
  W50 A_l = W100, so W50 = W100 B_l. B_l is the ridge least-squares solution over the five matrices stacked by rows,
  B_l = (W100^T W100 + lambda I)^{-1} W100^T W50. Each matrix is divided by the Frobenius norm of its 100% version (both
  checkpoints by the same factor), so the five weigh equally and a change of scale between checkpoints stays in B_l.
- **Output side, A_L.** The readout is logits = U diag(g) x / rms(x), with U the lm_head and g the final norm's gain. If the 50%
  head reads x50 as the 100% head reads x100, then U50 diag(g50) A_L = U100 diag(g100), so
  A_L = (M50^T M50 + lambda I)^{-1} M50^T M100 with M = U diag(g).
- lambda = 1e-4 x tr(W^T W) / 4096 of the matrix being inverted, fixed in advance, not tuned.
- Arithmetic in float64 on CPU; the lens is saved as float32, like the source, in the harness's `BaseLens` format.

## Variants, all fixed now

| name | lens | role |
|---|---|---|
| `wspV1_jpp_n64` | A_L J100_l B_l | **primary**: the edit |
| `wspV0_jpp_n64` | J100_l, unedited | control: the 100% lens used as is |
| `wspV2_jpp_n64` | A_L J100_l | ablation: output side only |
| `wspV3_jpp_n64` | J100_l B_l | ablation: input side only |

The ablations are reported, never promoted: the primary is V1 whatever the scores say.

## Scoring

The J++ post's evals in `safety-research/jpp_lens` at `d7efc0d`, driven by `jpp_evals/run_all.py --fits` of PR 18 (`3d8ccd6`), on
the 50% model (`olmo-3-7b-50pct`), on its `held-out:0` split, read from the 50% fit's own `model_correctness.csv` so the items are
the 198 the target was scored on. The same run scores the logit lens, the four variants and the target `olmo7b50pct_jpp_n64`, so
the target is re-scored beside them rather than quoted.

Primary metric: filtered macro recall@10 (Readout Filtering on, best rank over the seven layers, mean over the five tasks).
Uncertainty: a paired bootstrap over items, resampled within each task, 2000 resamples, seed 0.

## Verdicts, thresholds set now

1. **The edit helps**: V1 minus V0 is at least 2.0 points of filtered macro recall@10, and the 95% bootstrap interval of the
   difference lies above 0. Otherwise: no help (or harm, if the interval lies below 0).
2. **The edit matches the fitted lens**: V1 is within 2.0 points of the re-scored target, or above it. Otherwise: does not match.
3. **The 100% lens already transfers**: V0 is within 2.0 points of the re-scored target. Reported as context for 1 and 2.

Secondary, no verdicts: full-vocabulary recall@1 and @10, probe-swap top-1 and top-5, the per-task recalls, and every number again
on the held-out items that are also outside the 54 items the 100% lens's expert weights were fitted on (`fitting_item_names` in
`fits/olmo-3-7b/olmo7b_jpp_n64_stamp.json`), since the source lens has seen those.

## Caveats known now

- The 100% model adds mid-training, the long-context stage and YaRN rope scaling to pretraining; the 50% checkpoint has none of
  them, so the gap is more than training tokens.
- A J++ lens is a weighted mix of experts fitted on the 100% model's correctness split; the edit transports the mix, it does not
  refit the weights.
- A linear map between checkpoints is the simplest model of drift; the least-squares fit residuals are reported as a data-free
  diagnostic of how well it holds, and do not pick a variant.
