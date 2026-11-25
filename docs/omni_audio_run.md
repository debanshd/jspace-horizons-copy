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

## 1. Done (25 Nov): the lens checks

Run on 25 Nov. The 9B J-lens failed `check_lens` (deepest 0.353 against E4B's 0.486, and it loses to the logit lens): recorded
as a miss, it stays one. Qwen3-Omni-30B-A3B stopped at the startup picture check (picks ACD, letter mass 0.32 < 0.5). Kept here
for the record:

First, a clean-up. An earlier first pass, run before the end-token fix (`omni_defaults`), left `audio_clean` folders whose answers
ran to the token limit. They are void, and a later chain would reuse them, so move them aside. `check_lens_n250` generates
nothing, so its results stay valid; for Qwen3-Omni, which crashed at load, rerun it.

```bash
mkdir -p runs/void_noeos && for m in Qwen2.5-Omni-3B Qwen2.5-Omni-7B Qwen3-Omni-30B-A3B-Instruct; do [ -d runs/audio_clean/$m ] && mv runs/audio_clean/$m runs/void_noeos/; done
```

Then run the check:

```bash
scripts/lens_checks.sh 2>&1 | tee runs/lens_checks.log
```

It runs `check_lens` on every lens that is not Neuronpedia's own and would be read now:

- **Qwen3.5-9B's J-lens** (`--lens jlens`, PR #15), into `runs/check_lens`. This check decides: `scripts/video_sweep2.sh` reads the 9B through it only if it passes.
- **The Qwen omni models' 250-prompt lenses**, into `runs/check_lens_n250`, through `scripts/omni_first_pass.sh`, as described below.

For the omni models, `check_lens` runs on each model's 250-prompt lens, into `runs/check_lens_n250`, and nothing else: no audio experiment. A model
whose summary already exists there (the 3B, the 7B) is skipped, so this mainly runs Qwen3-Omni-30B-A3B. The check shows three
things:

- `Session` loads the thinker;
- it reads the picture through the full model;
- the lens is read at every read layer.

Its pass/fail decides nothing; the deciding lenses were the 1,000-prompt ones then, and are the 500-prompt ones since 25 Nov
(`config.DECIDING_LENS`).

Report, for each model:

- whether it loaded and read the picture;
- `deepest`, `beats_logit_lens` and `jlens` by layer from `runs/check_lens_n250/<model>/summary.json`.

If a model fails to load, report its `run.log`.

## 1b. Done (25 Nov): the calibration and the Qwen3-Omni diagnosis

Run on 25 Nov. Of Neuronpedia's Qwen lenses only the 2B passed the first rule (the 4B 0.312, the 9B's J++ 0.177, the 27B 0.264):
the rule is not calibrated for Qwen. Qwen3-Omni reads the picture; with no answer prefix 0.24–0.40 of its first token is on the
letters, with "(" 0.59–0.74. Kept here for the record:

The calibration: `check_lens` on Neuronpedia's own Qwen3.5-2B, 4B and 27B J-lenses and its 9B J++ lens, the lenses the original
Qwen results read, into `runs/check_lens_calibration`. What it decides is written in the script's header, before the run: if all
four pass, the rule stands for Qwen; if any fails, a new Qwen rule is written before any Qwen lens check is rerun.

```bash
scripts/lens_calibration.sh 2>&1 | tee runs/lens_calibration.log
```

The diagnosis: Qwen3-Omni-30B-A3B loaded as `Session` loads it, its picture check run as written (the failure recorded, not
raised), then the ten most likely next tokens at the answer position and its greedy answers. It measures and decides nothing.

```bash
uv run --frozen scripts/diagnose_preflight.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --run diagnose_preflight 2>&1 | tee runs/diagnose_preflight.log
```

Report:

- for each of the four calibration models, `deepest`, `beats_logit_lens`, `passes`, and `jlens` and `logit` by layer from
  `runs/check_lens_calibration/<model>/summary.json`;
- the whole of `runs/diagnose_preflight/Qwen3-Omni-30B-A3B-Instruct/summary.json`.

Then stop the pod.

## 1c. Done (25 Nov): the Qwen rule's rank calibration, and Qwen3-Omni's load

Run on 25 Nov. The Qwen rule came out void: Qwen3.5-9B's J++ lens ranks the model's top token worse than its own logit lens,
so the bar cannot reject a lens with no Jacobian, and the rule is retired. Qwen3-Omni-30B loads, picks "(" on card B (letter
mass 0.52) and reads the picture. Kept here for the record:

A first version of the Qwen rule (`38d96f9`) was withdrawn before it judged
anything: its bar was the weakest top-token agreement among the Neuronpedia Qwen lenses, 0.00 at most layers, so it passed any
lens. Move aside anything run with it:

```bash
mkdir -p runs/void_qwen_rule1 && for d in check_lens_qwen check_lens_qwen_n250; do [ -d runs/$d ] && mv runs/$d runs/void_qwen_rule1/; done
[ -d runs/check_lens_n250/Qwen3-Omni-30B-A3B-Instruct ] && mv runs/check_lens_n250/Qwen3-Omni-30B-A3B-Instruct runs/void_qwen_rule1/
```

The Qwen rule, second version, written before any lens is judged by it (`config.QWEN_CALIBRATION`, `scripts/check_lens.py`'s
docstring):

- **The measure:** where the model's top token ranks in the lens readout (median log10 rank), not whether it is the top one.
  The Neuronpedia Qwen lenses almost never make it the top one at mid depth.
- **The bar:** set mechanically from the four Neuronpedia Qwen lenses' ranks, measured now. Each of the four meets it.
- **Informative or void:** the bar must reject each of the four models' plain logit lens. Otherwise it is void, and no Qwen lens
  passes it.
- **New lenses only:** the 9B J-lens and the three 250-prompt omni lenses were seen before this rule and are refused
  (`config.SEEN_LENSES`); their first-rule results stand. The rule judges the 1,000-prompt lenses, once they exist.

The prefix change is also fixed: an omni model chooses its answer prefix on card B, which the picture check (A, C, D) never
shows, and the 0.5 bar stays.

```bash
scripts/lens_calibration.sh 2>&1 | tee runs/lens_calibration_ranks.log
```

```bash
MODELS=Qwen/Qwen3-Omni-30B-A3B-Instruct scripts/omni_first_pass.sh 2>&1 | tee runs/omni_first_pass2.log
```

Report:

- the last line of `runs/lens_calibration_ranks.log`: whether the Qwen rule is usable or void, and the four models'
  `rank_jlens` and `rank_logit` by layer printed above it;
- for Qwen3-Omni-30B, whether it loaded, the answer prefix it chose on card B, its picture check line from `run.log`, and
  `deepest`, `beats_logit_lens` and `jlens` by layer from `runs/check_lens_n250/Qwen3-Omni-30B-A3B-Instruct/summary.json`.
  These decide nothing.

Then stop the pod.

## 1d. Done (25 Nov): the lens validation's first checks

Check 1 passed (cosine ≥ 0.992); check 3 came out void (margins 2B +0.102, 4B −0.033, 27B −0.025). With the project owner's
approval, the deciding lenses are now the 500-prompt ones, and check 3 is a caveat that decides nothing (`config.LENS_CHECKS`).
Kept for the record:

The lenses fitted here are now validated by three checks written before they run (`config.LENS_CHECKS`): our recipe
reproduces Neuronpedia's Qwen3.5-2B lens, a lens's halves agree, and its concept readout is at least the weakest Neuronpedia
Qwen3.5 J-lens's. Run step 1 of `docs/lens_validation_run.md` (about 1 hour, about $4), and nothing else. It needs this branch's head, which
holds PR #12's code.

## 2. NOW: the 500-prompt lenses and their verdicts (about 3 hours on one H200, about $17)

Step 2 of `docs/lens_validation_run.md`, and nothing else. Its verdicts go to `runs/lens_validation/<model>/summary.json`.

---

**Later, not now:** steps 3 to 5 need a separate go-ahead.

## 3. (Later) Pin the 500-prompt lenses (a commit to PR #14, not on the pod)

Each model's `lens_file` becomes `<model>/n500/lens.pt`, with its sha256 and the new repo commit. Until then,
`scripts/audio_sweep.sh` refuses the omni models. It also refuses a model whose pinned lens has no passing verdict in
`runs/lens_validation`.

## 4. (Later) The audio chain on the deciding lenses

The two Qwen2.5-Omni models (an L40S is enough, but the H200 works too):

```bash
SWEEP=audio_sweep_static MODELS="Qwen/Qwen2.5-Omni-3B Qwen/Qwen2.5-Omni-7B" KEEP_GOING=1 scripts/audio_sweep.sh 2>&1 | tee runs/omni_chain_small.log
```

For each model, the chain runs these steps in order:

1. **Download:** the checkpoint and the deciding lens, its sha256 checked.
2. **The lens's verdict, `runs/lens_validation/<model>/summary.json`:** it must pass, for the lens that is pinned. If not, the model stops.
3. **`audio_clean`.**
4. **`audio_rules`.**
5. **Pilot:** the spoken / frozen / typed pilot, run twice and compared, then its gates. Then the chain stops for inspection.

Report for each model:

- `runs/lens_validation/<model>/summary.json`: the three checks and `passes`;
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

- **The lens fails validation:** our recipe does not reproduce Neuronpedia's lens, its halves disagree, or its concept readout is below the weakest Neuronpedia Qwen3.5 J-lens's. The fit reports (PR #13) showed a few such layers on WikiText. A failure means no audio run for that model. Report it; the rule stays as written.
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
   tar czf docs/omni_audio.tar.gz runs/check_lens runs/check_lens_n250 runs/check_lens_calibration* runs/lens_repro runs/lens_halves runs/lens_readout* runs/lens_validation runs/audio_clean runs/audio_rules runs/audio_sweep_static* runs/omni_chain_*.log runs/omni_first_pass.log
   ```

2. Write `docs/omni_audio_results.md` in the style of the audio stage 3 write-up, with numbers copied from the `summary.json` files. Give every interval's caveat beside it: 64 scenes reuse 15 recordings in one voice.
3. Open a results PR stacked on `claude/omni-models`.
4. Stop the pods.
