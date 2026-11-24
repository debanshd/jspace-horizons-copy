# The Qwen omni models on audio: how to run it

For whoever runs the pod. This runs the same chain as Gemma-4-E2B and E4B (audio stage 3), on Qwen2.5-Omni-3B and 7B and on
Qwen3-Omni-30B-A3B. Each uses the lens this project fitted for it (PR #13).

Every step's own gates decide whether the next step starts. **Do not edit a gate, a threshold or a script to make a step pass.** If a
model stops, write down where and why, and report it.

## 0. Set up (no GPU)

Use the audio worktree, `/workspace/jspace-audio`. Its `runs/check_lens/gemma-4-E4B-it/` is the reference the lens check needs; if
it is missing, the chain fits it again.

```bash
cd /workspace/jspace-audio
git fetch origin claude/omni-models
git checkout claude/omni-models
git merge --ff-only origin/claude/omni-models
```

The lenses are in the private Hugging Face repo `debanshd/jspace-omni-lenses`, so `.env` must hold an `HF_TOKEN` that can read it:

```bash
uv run --frozen python -c "from huggingface_hub import HfApi; from jspace.runs import load_env; load_env(); print(HfApi().model_info('debanshd/jspace-omni-lenses').sha)"
```

This must print `4648419f6346d3cbb356ed7b16b8a8b1fa2cbe75`.

## 1. Qwen2.5-Omni-3B and 7B, on one L40S (48 GB)

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B" KEEP_GOING=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_small.log
```

`KEEP_GOING=1` means a model that stops does not stop the next one. For each model, the chain runs these steps in order:

1. **Download:** the full checkpoint (12 and 22 GB) and the lens, its sha256 checked.
2. **`check_lens`:** the lens against the model, beside Gemma-4-E4B's Neuronpedia lens, under its rule as written. If it fails, the model stops.
3. **`audio_clean`:** the five questions, spoken in both orders and typed.
4. **`audio_rules`:** the three rule wordings.
5. **Pilot:** the 5-scene spoken / frozen / typed pilot, run twice and compared, then its gates. Then the chain stops for inspection.

Report for each model:

- `runs/check_lens/<model>/summary.json`: `passes`, `deepest`, `beats_logit_lens`;
- `runs/audio_clean/<model>/summary.json`: `recognition`, the `clean` table, `ready`;
- `runs/audio_rules/<model>/summary.json`: `diagnosis` and `passes`;
- the pilot gate's output.

Then run the full sweeps of the models whose pilots passed:

```bash
SWEEP=audio_sweep_static MODELS="<those models>" FULL=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_small_full.log
```

## 2. Qwen3-Omni-30B-A3B, on one H200 (141 GB)

Its text-generating part alone is about 60 GB in bf16, and the checkpoint is 70.5 GB. The volume and the steps are the same:

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen3-Omni-30B-A3B-Instruct" scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_30b.log
```

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen3-Omni-30B-A3B-Instruct" FULL=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_30b_full.log
```

Run the second command only if the pilot passed.

## What can stop a model, and what it means

- **`check_lens` fails:** the lens does not beat the plain logit lens at every read layer from half depth, or it falls more than 0.10 below E4B's lens at the deepest read layer. The fit reports (PR #13) showed a few such layers on WikiText. A failure means no audio run for that model. Report it; the rule stays as written.
- **`audio_clean` not ready:** the model does not name the recordings, or does not apply the say or animal rule, with the question first. Its audio is untestable by this design.
- **The pilot's controls fail on the spoken input:** stop and report. A frozen input the model does not recognize is untestable and does not stop the chain.

## 3. Compare with video, report, stop

None of these models was run on video, so the audio results stand alone; this comparison lists them as missing.

```bash
uv run --frozen scripts/compare_audio_video.py --audio runs/audio_sweep_static --video <the video worktree>/runs/video_sweep2 --out runs/audio_vs_video_omni.json
```

Write the report:

1. Pack the run folders:

   ```bash
   tar czf docs/omni_audio.tar.gz runs/check_lens runs/audio_clean runs/audio_rules runs/audio_sweep_static* runs/omni_chain_*.log
   ```

2. Write `docs/omni_audio_results.md` in the style of the audio stage 3 write-up, with numbers copied from the `summary.json` files. Give every interval's caveat beside it: 64 scenes reuse 15 recordings in one voice.
3. Open a results PR stacked on `claude/omni-models`.
4. Stop the pods.
