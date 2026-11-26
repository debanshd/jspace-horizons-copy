# Weight-space programming, experiment 2: rotation-only basis transport

Written 26 Nov 2026, before this edit is computed and before anything is scored. A new experiment, not a fix of experiment 1
(`RULE.md`), whose rule and outputs stay as they are.

## What was known when this was written

- Experiment 1's four lenses are frozen (`frozen.json`). None of them has been scored on the evals.
- After its freeze, the per-layer cosine similarity of experiment 1's lenses with the target was computed (`cosine.json`): on the
  seven read layers the unedited 100% lens has 0.644 with the target, the least-squares edit V1 0.577. That comparison is why this
  experiment exists. Cosine with the target plays no part in any verdict below, and no setting here was chosen to raise it.
- Experiment 1's diagnostics (`diagnostics.json`): the unconstrained least-squares maps explain only part of the change of weights
  (relative residual 0.58 to 0.68, against 0.68 to 0.90 for the identity), and their diagonals average about 0.7. The likely cause of
  V1 moving away is that an unconstrained least-squares map shrinks toward zero and mixes noise into every direction when most of
  the weight change is not a change of basis.

## The edit

The same model of drift as experiment 1, J50_l = A_L J100_l B_l, but each map is constrained to a rotation (orthogonal Procrustes),
so it can turn the basis and cannot shrink it or fold noise into it:

- B_l = argmin over orthogonal R of ||W100 R - W50||_F, the reading matrices of block l+1 stacked and scaled exactly as in
  experiment 1. With W100^T W50 = U S V^T, B_l = U V^T.
- A_L = argmin over orthogonal Q of ||M50 Q - M100||_F, M = lm_head diag(final norm gain). With M50^T M100 = U S V^T, A_L = U V^T.
- No scale factor: the readout normalises the transported residual, so a lens's overall scale does not matter.
- No ridge and no tuned setting. float64 on CPU, saved as float32 in the harness's format.

The inputs and the exclusions are those of experiment 1: the source lens and the two checkpoints' weights only.

## Variants, all fixed now

| name | lens | role |
|---|---|---|
| `wspP1_jpp_n64` | A_L J100_l B_l, both rotations | **primary** |
| `wspP2_jpp_n64` | A_L J100_l | ablation: output rotation only |
| `wspP3_jpp_n64` | J100_l B_l | ablation: input rotation only |
| `wspV0_jpp_n64` | J100_l unedited | control, experiment 1's file (sha256 `37392412…52b550`) |

## Scoring

One pod run scores experiment 1 and experiment 2 together, after both are frozen: the logit lens, V0 to V3, P1 to P3 and the target,
on the 50% model's `held-out:0` split from the 50% fit's own `model_correctness.csv`, with the J++ post's evals as in `RULE.md`.
Experiment 1's eval scores are not looked at before this experiment's lenses are frozen.

## Verdicts, thresholds set now (those of experiment 1)

Filtered macro recall@10, paired bootstrap over items within each task, 2000 resamples, seed 0:

1. **The edit helps**: P1 minus V0 is at least 2.0 points and its 95% interval lies above 0. Harms if the interval lies below 0.
2. **The edit matches the fitted lens**: P1 is within 2.0 points of the re-scored target, or above it.

Reported without a verdict: P1 minus V1 (rotation against least squares), the ablations, the secondary measures of `RULE.md`, and the
share of the weight change a rotation explains (data-free, from the weights).
