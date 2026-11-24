# Audio, stage 3: the full spoken / frozen / typed sweep, the 12B lens diagnostic, and the before/after check

PR #10 at `1891f64` (which includes `2d1243d`, the pilot gate following `audio_sweep_static`'s rule), run on 24 November 2026
(10:26 to 11:24 PM PST) on one L40S on the US-NC-1 volume, from the audio worktree (the video checkout untouched at `26657cd`),
following `docs/audio_stage3_run.md` (one change, below). Gemma-4-E2B and Gemma-4-E4B, question first. The rules are PR #10's,
written before the runs. Numbers are copied from each run's `summary.json`; every run folder and the chain log are in
`docs/audio_stage3.tar.gz` (sha256 `01b943c304aaed619201b79579d56e3a2372186297c5a67e401c6044e38a89ff`).

The 64 scenes reuse 15 recordings in one voice, so intervals over scenes are narrower than intervals over recordings.

## Step 1: the pilots under the fixed gate

Both pass (`pilot: passes`); the frozen word is "not perceived ... untestable, reported as a result; the chain goes on".

## Step 2: `audio_sweep_static`, full run (patching and steering)

| model (scenes) | input | say clean | animal clean | self-patch | every-layer donor patch | verdict | L_say / L_use | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-4-E2B (57) | spoken | 1.00 | 1.00 | 1.00 | 1.00 | together | 11 / 11 | **missed** | kept | kept | untestable | kept |
| | frozen | 0.00 | 0.51 | 1.00 | 0.00 | not perceived, untestable | | | | | | |
| | typed | 1.00 | 1.00 | 1.00 | 1.00 | together | 11 / 11 | **missed** | kept | kept | | |
| Gemma-4-E4B (64) | spoken | 1.00 | 1.00 | 1.00 | 1.00 | together | 19 / 19 | **missed** | kept | kept | untestable | kept |
| | frozen | 0.17 | 0.50 | 0.99 | 0.12 | not perceived, untestable | | | | | | |
| | typed | 1.00 | 1.00 | 1.00 | 1.00 | together | 19 / 19 | **missed** | kept | kept | | |

P4 (spoken against frozen) is untestable because the frozen word is not perceived; P5 (spoken against typed) is kept for both:
the spoken and the typed word give the same horizons. On the spoken and typed inputs every question's horizon is the same
window (E2B: say, animal, letter, set and ride all 11; E4B: all 19), so the name and the rule's use of it stop following the
donor together.

**Steering** (`media_steer`'s predictions 1 to 6): spoken, both models: 2 and 3 kept, 1, 4, 5 and 6 missed (as on video). Typed:
E2B 2, 3, 4 and 6 kept, 1 and 5 missed; E4B 2, 3 and 6 kept, 1, 4 and 5 missed. `steer_static_compare`: E2B's steering is the
same for spoken and frozen and for spoken and typed in every block; E4B's differs in the middle block (spoken against typed:
0.58 [0.44, 0.71]; spoken against frozen: 0.06 [-0.06, 0.19]).

Run: E2B 0.18 h, mean GPU 82%; E4B 0.35 h, 89%. No summary holds a NaN.

## Step 4: audio against video (`scripts/compare_audio_video.py`)

| model | spoken / moving video | frozen word / frozen video | typed word / still picture |
|---|---|---|---|
| Gemma-4-E2B | together 11 / 11 vs together 11 / 7 | untestable vs together 11 / 7 | together 11 / 11 vs untestable |
| Gemma-4-E4B | together 19 / 19 vs together 19 / 15 | untestable vs together 19 / 15 | together 19 / 19 vs together 19 / 15 |

The verdicts agree wherever both are testable ("together"), and L_say is the same; on audio the rule's use stops one window
later (L_use = L_say) than on video.

## Step 3: why Gemma-4-12B's lens fails (`scripts/diagnose_lens.py`, no pass/fail)

- **The check reads the 12B correctly.** The lens library's model prediction matches the model's own forward pass on 414 of
  416 positions (0.995; largest log-probability difference 1.70); E4B's: 416 of 416.
- **The 12B's prediction is formed at the very end.** The plain logit lens at layer 46 agrees with the model's prediction on
  0.5% of positions (median rank 87,128), against 8–9% at layers 41–44; the J-lens is best at layer 46 (0.139, median rank 848).
- **`layer_scalar`** (Gemma-4's per-layer output scale): the 12B's layer 0 is 0.053 and its layer 11 0.0045, as read from the
  weights before the run.

So the check measured the 12B's lens correctly; the lens is weak because this model's last layers rewrite its prediction. A
decision on the 12B needs a new lens check written before it runs.

## Step 5: AGENTS.md's before/after check of PR #10's shared-code changes

`video_sweep2 --n 4 --parts patch` on Gemma-4-E4B, from the same gate run (`runs/clip_naming3`) and data, at `26657cd` (run
`before_pr10`) and at `1891f64` (run `after_pr10`), compared with `scripts/compare_media.py --file video_sweep2.csv`:
**identical (2,160 rows)**.

One change to `docs/audio_stage3_run.md`: it compares the two runs with `scripts/check.py`, which compares only `results/`'s
fixed file list, so on these folders it finds none of them and still reports that everything matches. `compare_media.py`
compares each model folder's `video_sweep2.csv` row by row and fails if one is missing.

## Run

About 1 hour of L40S time (about $1.10); the pod stopped itself when the chain ended.
