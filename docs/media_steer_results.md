# media_steer: results on six models

Plan steps 6 and 7 of `plan.md` 3.2, run on 16 November 2026. The rule and the predictions were written before the runs; the code is `experiments/media_steer.py` at commit `ffb9d13`.

The pod is a RunPod RTX PRO 6000 Blackwell (96 GB) with a 300 GB network volume. Every number below comes from a run's `summary.json`. Rates are given as rate [95% bootstrap interval over the 48 held-out clips]. Outputs are in `runs/` on the pod and are not committed.

**Smoke check (plan step 5).** Both models ran twice, and the two runs are identical answer by answer (Qwen3.5-4B 2,840 rows, Gemma-4-E4B 4,680). Qwen3.5-0.8B, Qwen3.5-27B, Gemma-4-E2B and Gemma-4-31B also passed the same check in step 7.

## Step 6: Qwen3.5-4B and Gemma-4-E4B (`runs/media_steer`)

The 64 clips are those both models name.

| model / input | gates (say / animal) | dose | says Y: early / middle / late | knockout: last naming reader (depth) | ablation R1 / R1_nats (say) | predictions kept | missed | untestable |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B clip | 1.00 / 1.00 | 2.0 | 0.77 [0.65, 0.88] / 0.88 [0.77, 0.96] / 0.02 | 19 (0.59) | not needed / undetermined | 2, 3, 4 | 1, 5, 6, 7 | 8 (no audio) |
| Qwen3.5-4B static | 0.92 / 0.98 | 2.0 | 0.67 [0.52, 0.79] / 0.83 [0.73, 0.94] / 0.02 | 19 (0.59) | not needed / undetermined | 2, 3, 4 | 1, 5, 6, 7 | 8 |
| Gemma-4-E4B clip | 1.00 / 1.00 | 8.0 | 0.00 / 0.69 [0.56, 0.81] / 0.00 | 28 (0.67) | undetermined / **necessary** | 2, 7 | 1, 3, 4, 5, 6 | 8 (sound failed its gate) |
| Gemma-4-E4B static | 0.91 / 0.98 | 8.0 | 0.00 / 0.71 [0.58, 0.83] / 0.00 | 28 (0.67) | not needed / **necessary** | 2, 7 | 1, 3, 4, 5, 6 | |
| Gemma-4-E4B sound | **0.19 / 0.53** | 2.0 | not perceived, untestable | | | | | 1–8 |

## Step 7: all six models on one clip set (`runs/media_steer_all`, complete)

The 64 clips are those all six models name, so the set differs from step 6. Step 7 ran from 05:54 to 11:34 UTC.

| model / input | gates (say / animal) | dose | says Y: early / middle / late | knockout: last naming reader (depth) | ablation R1 / R1_nats (say) | predictions kept | missed | untestable |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B clip | 1.00 / 1.00 | 1.0 | 0.90 [0.81, 0.98] / 0.92 [0.83, 0.98] / 0.00 | 19 (0.79) | not needed / **necessary** | 1, 2, 3, 4 | 5, 6, 7 | 8 (no audio) |
| Qwen3.5-0.8B static | **0.89** / 0.97 | | not perceived, untestable (naming gate is 0.90) | | | | | 1–8 |
| Qwen3.5-4B clip | 1.00 / 1.00 | 2.0 | 0.79 [0.67, 0.90] / 0.90 [0.79, 0.98] / 0.02 | 19 (0.59) | not needed / undetermined | 2, 3, 4 | 1, 5, 6, 7 | 8 |
| Qwen3.5-4B static | 0.92 / 0.98 | 2.0 | 0.65 [0.50, 0.77] / 0.85 [0.75, 0.94] / 0.02 | 19 (0.59) | not needed / undetermined | 2, 3, 4 | 1, 5, 6, 7 | 8 |
| Qwen3.5-27B clip | 1.00 / 1.00 | 1.0 | 0.00 / 0.85 [0.75, 0.94] / 0.02 | 35 (0.55) | not needed / not needed | 2, 3, 6 | 1, 4, 5, 7 | 8 |
| Qwen3.5-27B static | 0.91 / 0.97 | 1.0 | 0.02 / 0.71 [0.58, 0.83] / 0.08 | 35 (0.55) | not needed / not needed | 2, 6 | 1, 3, 4, 5, 7 | 8 |
| Gemma-4-E2B clip | 1.00 / 1.00 | 8.0 (no dose calibrated) | 0.00 / 0.00 / 0.00 | 32 (0.91) | undetermined / undetermined | 2, 3, 7 | 1, 4, 5, 6 | 8 (sound failed its gate) |
| Gemma-4-E2B static | **0.81** / 0.98 | | not perceived, untestable | | | | | 1–8 |
| Gemma-4-E2B sound | **0.09 / 0.52** | | not perceived, untestable | | | | | 1–8 |
| Gemma-4-E4B clip | 1.00 / 1.00 | 8.0 | 0.00 / 0.69 [0.56, 0.81] / 0.00 | 28 (0.67) | not needed / **necessary** | 2, 7 | 1, 3, 4, 5, 6 | 8 (sound failed its gate) |
| Gemma-4-E4B static | 0.91 / 0.98 | 8.0 | 0.00 / 0.73 [0.60, 0.85] / 0.00 | 28 (0.67) | not needed / **necessary** | 2, 7 | 1, 3, 4, 5, 6 | |
| Gemma-4-E4B sound | **0.22 / 0.53** | 2.0 | not perceived, untestable | | | | | 1–8 |
| Gemma-4-31B clip | 1.00 / 1.00 | 8.0 (no dose calibrated) | 0.02 / 0.00 / 0.00 | 32 (0.53) | not needed / not needed | 2, 7 | 1, 3, 4, 5, 6 | (no audio input) |
| Gemma-4-31B static | 0.94 / 0.97 | 8.0 (no dose calibrated) | 0.02 / 0.00 / 0.00 | 32 (0.53) | not needed / not needed | 2, 7 | 1, 3, 4, 5, 6 | |

"No dose calibrated" means no dose in the calibration set steered naming, so the run used the largest dose, 8.0.

Across the six models:

- **Naming steered from the media tokens.** It works from the middle block on Qwen3.5-0.8B, 4B and 27B and on Gemma-4-E4B (0.69–0.92). It **fails on Gemma-4-E2B and Gemma-4-31B**: no block steers naming even at dose 8.0 (0.00–0.02). The random direction steers nothing anywhere (≤ 0.02), and the late block's push goes through the last token, not the media tokens.
- **The rule question.** Where naming steers, the second question mostly does not follow Y ("only said"), or the result is undetermined. On Gemma-4-E4B and 31B the early and middle blocks move the rule answer by 0.27–0.50 without moving the name.
- **Steering from the early block.** It works only on the small Qwen models (0.8B 0.90, 4B 0.79) and fails on every other model (0.00–0.02). That is why prediction 1 is kept only on Qwen3.5-0.8B's clip.
- **Prediction 5 misses everywhere.** The knockout's last naming read is at 0.53–0.91 of depth, never inside 0.67–0.75 with the steering edge agreeing.
- **Prediction 6.** It is kept only on Qwen3.5-27B, where removing the lens part costs naming 0.00–0.02. Elsewhere that removal costs 0.15–0.71.
- **Prediction 7.** It misses on every Qwen model, where the clip differs from the static video in one block (early on the 4B, middle on the 0.8B and the 27B). It is kept on Gemma-4-E4B and 31B, where clip and static agree ("nothing temporal tested").
- **Prediction 8 is untestable on every model.** Qwen3.5 and Gemma-4-31B take no audio, and Gemma-4-E2B and E4B fail the sound gate (naming 0.09 and 0.22), as in RECORD row 6.
- **Inputs that failed their gates:** Qwen3.5-0.8B static (0.89) and Gemma-4-E2B static (0.81), both against a naming gate of 0.90.

Draft `docs/RECORD.md` rows are written but **not committed**; they wait for review.

The run outputs (`runs/`, 39 MB packed) are not committed, following AGENTS.md. They are on the pod's network volume and in an archive handed over outside git (sha256 `b2a1e4bd899a7e68a56e125a4ac13737fc39cc2e5e91ed10349f1beefe0a7075`).

## Run

| run | wall time | mean / median GPU utilization | GPU-busy share | batch rows | encoder cache |
|---|---|---|---|---|---|
| step 6 Qwen3.5-4B | 787 s | 91.1% / 100% | 0.87 | 512 | off (\|d log p\| 0.20–0.30) |
| step 6 Gemma-4-E4B | 1,993 s | 98.0% / 100% | 0.85 | 253 / 247 video, 512 sound | on for video |
| step 7 Qwen3.5-0.8B | 292 s | 86.0% / 96% | 0.78 | 512 | off |
| step 7 Qwen3.5-4B | 774 s | 91.9% / 100% | 0.87 | 512 | off |
| step 7 Qwen3.5-27B | 3,007 s | 98.2% / 100% | 0.93 | 149 / 142 (halved once on out-of-memory) | on |
| step 7 Gemma-4-E2B | 2,083 s | 98.1% / 100% | 0.89 | 272 / 266 video, 512 sound | off (\|d log p\| 0.69–0.76) |
| step 7 Gemma-4-E4B | 1,989 s | 98.7% / 100% | 0.85 | 253 / 247 video, 512 sound | on for video |
| step 7 Gemma-4-31B | 8,297 s | 99.2% / 100% | 0.95 | 19–39 (halved on out-of-memory) | on |

Media tokens: 196 video on Qwen; **512 video on Gemma (8 × 64, not the plan's 560)**; 250 audio.
