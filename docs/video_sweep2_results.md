# video_sweep2: results on the four smaller models

PR #8's `video_sweep2` (`26657cd`), run on 18–19 November 2026 (11:55 PM to 6:39 AM PST), one model per L40S, with the
pinned model revisions and the matched set at `e33c6ae8`. Each model ran on its gated schedule: `clip_naming4` for
Qwen3.5-0.8B (run first on its pod), `clip_naming3` for the others. The design, controls and rule are PR #8's (`video_sweep`'s
rule, unchanged), written before the run. Every number is copied from a run's `summary.json`. Each model's runs (the smoke
check twice, the full run, and for Qwen3.5-0.8B its `clip_naming4` run) and its chain log are in `docs/video_sweep2_<tag>.tar.gz`:

| model | archive | sha256 |
|---|---|---|
| Qwen3.5-0.8B | `video_sweep2_q08.tar.gz` | `0e1ed2bd759a3b10fe1fe6c937973635bf7c0d444416768a1f6a1300fbfa9f42` |
| Gemma-4-E2B | `video_sweep2_e2b.tar.gz` | `6f35c2ee055ce17a7b9f4c3e333f1d7195d2e8a1d0835182c5768c3194babf92` |
| Qwen3.5-4B | `video_sweep2_q4.tar.gz` | `75092b4318414fcd52dcebd44261176efc09f07e4a00d6cdce6d58598985eebe` |
| Gemma-4-E4B | `video_sweep2_e4b.tar.gz` | `71e6147756e4ace626d7fc2dda1bd4b010c63311f3a867e4756704d8814913f9` |

**Smoke checks.** Each model's 5-scene smoke check, run twice, is identical answer for answer (0.8B 5,700 rows, E2B 6,570,
4B 6,510, E4B 7,380).

**`clip_naming4`** (Qwen3.5-0.8B, `clip_naming`'s question with `clip_naming3`'s rule): named 0.918 / 0.849 / 0.890 of the
73 clips (moving / frozen / still); 54 moving-input scene pairs (PR #6's floor was 52); 4% of moving and 11% of frozen
answers hedged and were asked again.

## Patching (each input: naming gate, controls, verdict)

| model (scenes) | input | say clean | every-layer patch | verdict | L_say / L_use | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (54) | moving | 0.85* | 0.85 | untestable | | | | | | |
| | frozen | 0.89* | 0.87 | untestable | | | | | | |
| | still | 0.96 | 0.96 | never | 15 / none | untestable | untestable | kept | | |
| Gemma-4-E2B (50) | moving | 0.96 | 1.00 | together | 11 / 7 | **missed** | kept | kept | kept | untestable |
| | frozen | 0.92 | 0.92 | together | 11 / 7 | **missed** | kept | kept | | |
| | still | 0.84* | 0.92 | untestable | | | | | | |
| Qwen3.5-4B (59) | moving | 1.00 | 1.00 | too late | 19 / 7 | kept | kept | kept | kept | kept |
| | frozen | 0.97 | 1.00 | too late | 19 / 7 | kept | kept | kept | | |
| | still | 0.97 | 1.00 | too late | 19 / 7 | kept | kept | kept | | |
| Gemma-4-E4B (57) | moving | 1.00 | 1.00 | together | 19 / 15 | **missed** | kept | kept | kept | kept |
| | frozen | 0.96 | 0.96 | together | 19 / 15 | **missed** | kept | kept | | |
| | still | 0.98 | 0.98 | together | 19 / 15 | **missed** | kept | kept | | |

`*` fails the 0.90 naming gate. Self-patching gives the clean answer on 0.96–1.00 of rows everywhere. On Qwen3.5-0.8B's
still input the animal rule fails its gate (0.72), so `L_use` is not read. P4 and P5 are scored on the moving input only.

The verdicts repeat `media_patch`'s on the VGGSound clips (PR #5): Qwen "too late", Gemma "together",
so P1 is kept on Qwen3.5-4B and missed on both Gemma models.

## Steering (`media_steer`'s rule, predictions 1–6)

On every testable input of Gemma-4-E2B, Qwen3.5-4B and Gemma-4-E4B, predictions 2 and 3 are kept and 1, 4, 5 and 6 are
missed, except Gemma-4-E4B's frozen input, where 3 is missed too. Qwen3.5-0.8B's still input: 2 and 3 kept, 1, 5 and 6
missed, 4 untestable; its moving and frozen inputs are untestable.

## The answers (read from each `video_sweep2.csv`)

- **Qwen3.5-0.8B**, clean `say` misses: "animal" for cats and horses (4 moving, 3 frozen), "food" for a pizza, and lists it
  repeated when asked again ("bicycle, electric", "Bicycle, motor"). Of its 1,099 `say` rows asked again after a hedge, 46%
  changed to one name ("laptop;computer" → "laptop") and 54% repeated the list.
- **Gemma-4-E2B**: "Decoration" for a clock (moving, frozen); on the still picture horses named as dogs ("Canine",
  "Canis lupus familiaris") and a train as "railway".
- **Qwen3.5-4B**: only "sportscar" for a car (frozen and still, 4 answers), one word that the scoring does not read as car.
- **Gemma-4-E4B**: "bowl" for a cup and "dog" for a bird (3 answers in all).

## Run

| model | full run | mean GPU |
|---|---|---|
| Qwen3.5-0.8B | 0.9 h | 77% |
| Gemma-4-E2B | 2.9 h | 93% |
| Qwen3.5-4B | 3.9 h | 93% |
| Gemma-4-E4B | 5.6 h | 96% |

About 18 L40S pod-hours in all (smoke checks and loading included), about $20. Each pod stopped itself when its chain ended.
`torch.linalg.svd` warned once on Qwen3.5-4B and once on Gemma-4-E4B that a batch did not converge (the per-scene principal
direction of the clean states); no summary holds a NaN.
