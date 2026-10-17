# jspace-horizons

Experiments that read, patch, ablate and steer the image-token states of vision-language models (Qwen3.5, Gemma-4) with the [J-lens](https://arxiv.org/abs/2607.15495). The repository has the code, the data and model pins, and the outputs of the original runs, so every result can be reproduced and checked.

![Residual-stream patching across depth](results/examples/ex1_depth_one_scene.png)
*Residual-stream patching across depth: a donor's image-token states written into a receiver, four layers at a time (`depth_sweep`).*

![Activation steering with a word's J-lens direction](results/examples/ex5_written_name.png)
*Activation steering with a word's J-lens direction (`name_steering`).*

![Directional ablation on photos with written words](results/examples/ex7_written_word.png)
*Directional ablation of a written word's J-lens direction (`printed_word`, `scam_notes`).*

More figures, every input at full size, and every answer shown with its prompt are in [`results/examples/`](results/examples).

## Quickstart

Requirements: Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/), and the DejaVu fonts (`apt install fonts-dejavu-core`).

```bash
git clone https://github.com/the-infiltrator/jspace-horizons && cd jspace-horizons
uv sync                       # Python 3.12 and the exact package versions (uv.lock)
uv run scripts/download.py    # data, Qwen3.5-4B and its lens at pinned revisions, checked by sha256 (about 11 GB)
scripts/reproduce.sh          # the Qwen3.5-4B runs, the figures, and the comparison with results/
```

To try one experiment on four scenes (a few minutes):

```bash
uv run experiments/depth_sweep.py --n 4 --run try_depth_sweep
```

## Experiments

| script | what it does |
|---|---|
| `depth_sweep` | patches a donor's image-token states into a receiver one four-layer window at a time, from the input to the last layer |
| `back_patching` | writes the same states at earlier or later layers, and restores the receiver's own states in between |
| `subspace_patching` | patches only part of the state: the lens subspace, its complement, or random subspaces |
| `name_steering` | removes or adds a word's lens direction on an object's image tokens |
| `gqa_ablation` | ablates lens subspaces and single lens directions from GQA photos |
| `printed_word` | ablates the lens direction of a word printed on an object |
| `scam_notes` | the same on SCAM photos with a hand-written note, plus steering |

Each script writes `runs/<run>/<model>/`: a CSV of every answer, `summary.json` (estimates, 95% bootstrap intervals, verdicts), `stimuli.png`, `config.json` and `run.log`. `--help` gives each design; [RESEARCH.md](RESEARCH.md) gives the method and controls.

## Reproducibility

`scripts/reproduce.sh` downloads what is missing, runs the ten experiment runs, draws the figures into `runs/examples/`, and runs `scripts/check.py`, which compares every answer, verdict and number with `results/`. Finished runs are skipped on a restart. Data goes to `data/` and models to the Hugging Face cache (`HF_HOME`); neither is part of the repository. Gemma-4-31B is opt-in: `GEMMA=1 scripts/reproduce.sh` (or `download.py --gemma`) adds its 69 GB download and two runs; without it the figures take its outputs from `results/`.

| | runs | time (RTX PRO 6000, 96 GB) | GPU memory |
|---|---|---|---|
| Qwen3.5-4B | 8 | about 1 hour | up to 50 GB |
| Gemma-4-31B | 2 | about 45 minutes | up to 88 GB |

An exact match needs a GPU with at least 80 GB, since batch sizes follow GPU memory; on another GPU type a few answers can change, and `check.py` lists each one.

| pinned | |
|---|---|
| packages | `uv.lock` (torch 2.8.0+cu128, transformers 5.19.0, flash-linear-attention 0.5.2) |
| J-lens code | [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) at `581d398` |
| models | `Qwen/Qwen3.5-4B` at `851bf6e`, `google/gemma-4-31B-it` at `842da37` |
| lenses | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) at `e166deb`, sha256-checked |
| data | COCO 2017 val, `BLISS-e-V/SCAM` at `8c674f5`, `lmms-lab-encoder/GQA` at `a6e72d6`, sha256-checked |
| judge | `gemini-2.5-flash`; the original verdicts in `results/judge_cache.json`, so no API key is needed when answers match |

## Layout

```
src/jspace/     shared code: model and lens session, lens readouts, interventions, stimuli, judge, statistics, run folders
experiments/    one script per experiment
scripts/        download.py, reproduce.sh, make_figures.py, check.py
results/        outputs of the original runs (13-14 Oct 2026), the reference for check.py
tests/          CPU tests: uv run pytest
```

[AGENTS.md](AGENTS.md) has the rules for changing the code; [TASK.md](TASK.md) the current task.

## Credits

The J-lens is from [Gurnee et al. (2026)](https://arxiv.org/abs/2607.15495) and [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache 2.0); the fitted lenses are Neuronpedia's. Models: [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) and [Gemma 4](https://huggingface.co/google/gemma-4-31B-it), under their own licences. Data: [COCO](https://cocodataset.org) (annotations CC BY 4.0), [SCAM](https://huggingface.co/datasets/BLISS-e-V/SCAM) (CC BY-NC 4.0), [GQA](https://cs.stanford.edu/people/dorarad/gqa/) (CC BY 4.0).
