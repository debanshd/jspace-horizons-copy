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

## 1. NOW: `check_lens` only (one H200, about 15 minutes)

**This is the only step to run now.** Steps 2 to 5 below come later, and only with a separate go-ahead.

First, a clean-up. An earlier first pass, run before the end-token fix (`omni_defaults`), left `audio_clean` folders whose answers
ran to the token limit. They are void, and a later chain would reuse them, so move them aside. `check_lens_n250` generates
nothing, so its results stay valid; for Qwen3-Omni, which crashed at load, rerun it.

```bash
mkdir -p runs/void_noeos && for m in Qwen2.5-Omni-3B Qwen2.5-Omni-7B Qwen3-Omni-30B-A3B-Instruct; do [ -d runs/audio_clean/$m ] && mv runs/audio_clean/$m runs/void_noeos/; done
```

Then run the check:

```bash
scripts/omni_first_pass.sh 2>&1 | tee runs/omni_first_pass.log
```

It runs `check_lens` on each model's 250-prompt lens, into `runs/check_lens_n250`, and nothing else: no audio experiment. A model
whose summary already exists there (the 3B, the 7B) is skipped, so this mainly runs Qwen3-Omni-30B-A3B. The check shows three
things:

- `Session` loads the thinker;
- it reads the picture through the full model;
- the lens is read at every read layer.

Its pass/fail decides nothing; the deciding lenses are the 1,000-prompt ones (`config.DECIDING_LENS`).

Report, for each model:

- whether it loaded and read the picture;
- `deepest`, `beats_logit_lens` and `jlens` by layer from `runs/check_lens_n250/<model>/summary.json`.

If a model fails to load, report its `run.log`.

---

**Later, not now:** steps 2 to 5 need a separate go-ahead.

## 2. (Later) Extend the lenses to 1,000 prompts (PR #12's code; one H200, about 9.4 hours, about $50)

Run this in the worktree that holds `runs/fit_lens` (the three 250-prompt fits), at PR #12's head:

```bash
EXTEND_TO=1000 PUSH_TO=debanshd/jspace-omni-lenses scripts/fit_lenses.sh 2>&1 | tee runs/fit_extend.log
```

Each lens is uploaded under `<model>/n1000/`. Report:

- the repo's new commit;
- each `runs/fit_lens_n1000/<model>/meta.json`: `n_prompts`, `lens_sha256`, and the merged parts.

## 3. (Later) Pin the 1,000-prompt lenses (a commit to PR #14, not on the pod)

Each model's `lens_file` becomes `<model>/n1000/lens.pt`, with its sha256 and the new repo commit. Until then,
`scripts/audio_sweep.sh` refuses the omni models.

## 4. (Later) The audio chain on the deciding lenses

The two Qwen2.5-Omni models (an L40S is enough, but the H200 works too):

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B" KEEP_GOING=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_small.log
```

For each model, the chain runs these steps in order:

1. **Download:** the checkpoint and the deciding lens, its sha256 checked.
2. **`check_lens`, into `runs/check_lens`:** the deciding check. If it fails, the model stops.
3. **`audio_clean`.**
4. **`audio_rules`.**
5. **Pilot:** the spoken / frozen / typed pilot, run twice and compared, then its gates. Then the chain stops for inspection.

Report for each model:

- `runs/check_lens/<model>/summary.json`: `passes`, `deepest`, `beats_logit_lens`;
- `runs/audio_rules/<model>/summary.json`: `diagnosis` and `passes`;
- the pilot gate's output.

Then run the full sweeps of the models whose pilots passed:

```bash
SWEEP=audio_sweep_static MODELS="<those models>" FULL=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_small_full.log
```

Qwen3-Omni-30B-A3B needs an H200, since its text-generating part alone is about 60 GB in bf16:

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen3-Omni-30B-A3B-Instruct" scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_30b.log
```

If its pilot passed:

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen3-Omni-30B-A3B-Instruct" FULL=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_30b_full.log
```

## What can stop a model, and what it means

- **`check_lens` fails:** the lens does not beat the plain logit lens at every read layer from half depth, or it falls more than 0.10 below E4B's lens at the deepest read layer. The fit reports (PR #13) showed a few such layers on WikiText. A failure means no audio run for that model. Report it; the rule stays as written.
- **`audio_clean` not ready:** the model does not name the recordings, or does not apply the say or animal rule, with the question first. Its audio is untestable by this design.
- **The pilot's controls fail on the spoken input:** stop and report. A frozen input the model does not recognize is untestable and does not stop the chain.

## 5. (Later) Compare with video, report, stop

None of these models was run on video, so the audio results stand alone; this comparison lists them as missing.

```bash
uv run --frozen scripts/compare_audio_video.py --audio runs/audio_sweep_static --video <the video worktree>/runs/video_sweep2 --out runs/audio_vs_video_omni.json
```

Write the report:

1. Pack the run folders:

   ```bash
   tar czf docs/omni_audio.tar.gz runs/check_lens runs/check_lens_n250 runs/audio_clean runs/audio_rules runs/audio_sweep_static* runs/omni_chain_*.log runs/omni_first_pass.log
   ```

2. Write `docs/omni_audio_results.md` in the style of the audio stage 3 write-up, with numbers copied from the `summary.json` files. Give every interval's caveat beside it: 64 scenes reuse 15 recordings in one voice.
3. Open a results PR stacked on `claude/omni-models`.
4. Stop the pods.
