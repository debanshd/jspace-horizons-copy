# Audio stage 3: how to run it

For whoever runs the pod. Use one L40S (48 GB) on the US volume, in the audio worktree that stages 1 and 2 used
(`/workspace/jspace-audio`). Every command runs from that folder. Each step's own gates decide whether the next step starts, so
**do not edit a gate, a threshold or a script to make a step pass**. If a step stops, write down why and report it.

## 0. Update the code (2 minutes, no GPU)

```bash
cd /workspace/jspace-audio
git fetch origin claude/audio-sweep
git checkout claude/audio-sweep
git merge --ff-only origin/claude/audio-sweep
git log --oneline -3
```

The log must include `2d1243d` (the pilot gate fix) and the commit that adds `scripts/diagnose_lens.py`.

## 1. Re-check the two pilots under the fixed gate (1 minute, no GPU)

```bash
uv run --frozen scripts/audio_gate.py pilot runs/audio_sweep_static_pilot/gemma-4-E2B-it/summary.json
uv run --frozen scripts/audio_gate.py pilot runs/audio_sweep_static_pilot/gemma-4-E4B-it/summary.json
```

**Expected:**

- both end in `pilot: passes`;
- the line `q_frozen: not perceived ... untestable` appears.

If either ends in `FAILS`, stop and report its output.

## 2. The full spoken / frozen / typed sweep for Gemma-4-E2B and E4B (GPU, the long step)

```bash
SWEEP=audio_sweep_static MODELS="google/gemma-4-E2B-it google/gemma-4-E4B-it" FULL=1 scripts/audio_sweep.sh 2>&1 | tee runs/audio_stage3_chain.log
```

The chain skips what already exists (`audio_clean`, `audio_rules`, both pilots), repeats the pilot comparison and gates, then runs
the full sweep for each model: patching and steering, on the question-first schedule.

- **Gated scenes:** 57 for E2B, 64 for E4B.
- **Run time:** not measured. For scale, `video_sweep2`'s E2B took about 3 hours on video prompts of 2,112 tokens. Audio prompts are about 120 tokens, but the donor capture covers five questions.
- **Watch** `runs/audio_sweep_static/<model>/gpu.csv`. Utilization should sit high during the sweeps; the low-GPU stretches are the per-prompt steps.
- **Expected result:** `q_frozen` ends "not perceived, untestable" (its P4 untestable), as the pilots showed. This is a result, not a failure.

## 3. The Gemma-4-12B lens diagnostic (GPU, about 10 minutes; before or after step 2)

```bash
uv run --frozen scripts/diagnose_lens.py --model google/gemma-4-E4B-it --run diagnose_lens
uv run --frozen scripts/diagnose_lens.py --model google/gemma-4-12B-it --run diagnose_lens \
  --extra-lens lamm-mit/gemma4-jacobian-lenses:gemma4-12b-it/paper/seed0.pt@c095b094c52dd7afeb1310bb97ea35dc5005e12f
```

It has no pass/fail. Report these from each model's `runs/diagnose_lens/<model>/summary.json` and `diagnose.csv`:

- **The `target:` line:** the share of positions where the lens library's model prediction equals the model's own forward pass. Expected near 1.00; anything lower means our check reads the model wrongly. Also report `error`, if present.
- **Per layer:** the J-lens and logit-lens top-1 agreement, the median rank and the residual norm. Note especially layers 44–47 of the 12B against layers 38–41 of E4B.
- **`layer_scalar`:** as weights the run reads them. From the weights read beforehand, the 12B's last layer is 0.048 and E4B's 0.445.

Do **not** start any 12B audio run, and do not change `scripts/check_lens.py`. Whether the 12B goes ahead is decided from this report,
by a new lens check written before it runs.

## 4. Compare audio with video (1 minute, no GPU)

```bash
uv run --frozen scripts/compare_audio_video.py --audio runs/audio_sweep_static --video <the video worktree>/runs/video_sweep2 --out runs/audio_vs_video.json
```

## 5. Report and stop

1. Pack the results, then write their sha256 into the report:

   ```bash
   tar czf docs/audio_stage3.tar.gz runs/audio_sweep_static runs/audio_sweep_static_pilot* runs/diagnose_lens runs/audio_vs_video.json runs/audio_stage3_chain.log
   ```

2. Write `docs/audio_stage3_results.md` in the style of stages 1 and 2, with the numbers copied from the `summary.json` files:
   - per model and input: the verdict, `L_say` / `L_use`, and P1–P5;
   - `static_compare` and `steer_static_compare`;
   - controls;
   - the step 3 diagnostic;
   - next to every interval: the 64 scenes reuse 15 recordings in one voice, so intervals over scenes are narrower than intervals over recordings.
3. Push both files to `claude/audio-results` (PR #11).
4. Stop the pod.

## Before PR #10 is merged (AGENTS.md)

PR #10 changed shared code: `src/jspace/batched.py` (`Span`, `hit_limit`), `model.py` and `config.py` (the lens source). AGENTS.md
asks for each affected experiment to be run at small size before and after the change, and the two run folders compared.

1. Run `video_sweep2` with `--n 4 --parts patch` on Gemma-4-E4B twice, from the same gate run:
   - at `26657cd`, as run `before_pr10`;
   - at PR #10's head, as run `after_pr10`.
2. Compare the two:

   ```bash
   uv run --frozen scripts/check.py --runs runs/after_pr10 --reference runs/before_pr10
   ```

The outputs must be identical.
