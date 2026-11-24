# Mid-size video models (PR #15): how to run them

For whoever runs the pod. It adds Qwen3.5-2B and Qwen3.5-9B to `video_sweep2`, between the 0.8B and 4B and between the 4B and
27B already run (PR #9), so the Qwen sizes are 0.8B, 2B, 4B, 9B and 27B. Both read a J-lens:

| model | lens |
|---|---|
| Qwen3.5-2B | its pinned Neuronpedia J-lens |
| Qwen3.5-9B | `bcywinski/jacobian-lens-qwen3.5-9b`: 1,000 prompts, Neuronpedia's recipe for the 9B base model (`--lens jlens`, `config.ALT_LENSES`) |

The 9B's pinned lens stays its Neuronpedia J++ lens, because the results tables read it.

Use one L40S (48 GB), in the video worktree that ran PR #9 (its `runs/` holds the 0.8B and 4B runs). **Do not edit a gate, a
threshold or a script to make a step pass.**

## 0. Update the code

```bash
git fetch origin claude/mid-video
git checkout claude/mid-video
git merge --ff-only origin/claude/mid-video
```

## 1. The naming gate (`clip_naming3`, the same rule as the 4B's and the Gemmas')

```bash
MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B" scripts/clip_naming3.sh 2>&1 | tee runs/mid_naming.log
```

Report each model's `runs/clip_naming3/<model>/summary.json`:

- the share named per input;
- `gated_scenes` (moving).

## 2. The smoke check twice, then the full sweep

```bash
MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-9B" scripts/video_sweep2.sh 2>&1 | tee runs/mid_sweep.log
```

For each model, this runs:

1. **Download:** the model and its lens. For the 9B, `--lens jlens` is downloaded and checked against its sha256.
2. **Smoke check:** the 5-scene check twice, compared answer by answer.
3. **Full sweep:** the full `video_sweep2`, patching and steering, on the moving, frozen and still inputs.

Each run's `config.json` names the lens it read (`lens_repo`, `lens_file`). The 9B's must be `bcywinski/jacobian-lens-qwen3.5-9b`
and `lens_n1000.pt`.

Run time is not measured. As a guide, the 4B's full run fit on one L40S, the 2B should be shorter and the 9B about twice as long.

## 3. Report and stop

1. Pack the run folders:

   ```bash
   tar czf docs/video_sweep2_mid.tar.gz runs/clip_naming3/Qwen3.5-2B runs/clip_naming3/Qwen3.5-9B runs/video_sweep2*/Qwen3.5-2B runs/video_sweep2*/Qwen3.5-9B runs/mid_*.log
   ```

2. Write the results, per input: the verdict, `L_say` / `L_use`, P1–P5, the controls and the steering predictions. Write them beside the 0.8B, 4B and 27B rows of PR #9's table.
3. Open a results PR stacked on `claude/mid-video`.
4. Stop the pod.
