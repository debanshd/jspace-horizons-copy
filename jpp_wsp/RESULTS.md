# Results: experiments 1 and 2, scored together (26 Nov 2026, one H200, `out/`)

Setup check: the fitted 50% lens re-scored at 39.5 filtered recall@10, the number of the original run, on the same 198 held-out
items (multihop 12, multilingual 13, typo 77, association 77, poetry 19).

| lens | filtered r@10 | filtered r@1 | full r@10 | probe-swap top-1 |
|---|---|---|---|---|
| logit | 28.1 | 8.9 | 28.1 | 9.8 |
| target: fitted 50% J++ | 39.5 | 21.2 | 38.9 | 48.8 |
| V0: 100% lens unedited | 36.1 | 17.2 | 34.7 | 51.2 |
| **V1: least-squares transport (exp 1 primary)** | **46.2** | 26.6 | 43.5 | 39.0 |
| V2: output side only | 49.1 | 22.5 | 47.0 | 43.9 |
| V3: input side only | 35.5 | 17.4 | 34.7 | 48.8 |
| **P1: rotation transport (exp 2 primary)** | **40.8** | 24.4 | 39.8 | 39.0 |
| P2: output rotation only | 42.3 | 21.5 | 40.6 | 46.3 |
| P3: input rotation only | 34.7 | 17.3 | 34.2 | 48.8 |

Paired bootstrap of filtered r@10 (points, 95% interval), `verdicts1.json` and `verdicts2.json`:

- Experiment 1: V1 - V0 = +10.1 [4.9, 15.7]: **the edit helps**. V1 - target = +6.7 [1.4, 12.6]: **matches the fitted lens** (above it).
- Experiment 2: P1 - V0 = +4.7 [0.7, 9.3]: **the edit helps**. P1 - target = +1.3 [-3.6, 6.4]: **matches the fitted lens**.
- V0 - target = -3.4 [-7.3, 0.4]: the 100% lens unedited does not transfer (just over the 2-point margin).
- Without a verdict: P1 - V1 = -5.4 [-11.5, 0.5], so the rotation did not beat least squares. The output side carries the gain in both
  experiments (V2 49.1, P2 42.3), and the input side alone does nothing (V3, P3 at V0's level). On the held-out items the 100% lens's
  expert weights never saw, the order is the same (V2 49.5, V1 43.9, P1 39.7, target 39.2, V0 35.5).

What this does not show:
- The causal eval goes the other way: on probe-swap the edits are below the target and V0 (V1 and P1 39.0% against 48.8% and 51.2%,
  41 trials).
- Cosine with the target (`cosine.json`) did not predict the readout: V1 is further from the target than V0 by cosine and scores
  10 points higher.
- Multihop, multilingual and poetry have 12 to 19 items each, so their per-task numbers move a lot with a few items. Most of the gain
  comes from typo and association.
