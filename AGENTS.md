# AGENTS.md

How this repository is organized and the rules for changing it, for coding agents and contributors. [RESEARCH.md](RESEARCH.md) describes the method; [TASK.md](TASK.md) the current task.

## Layout

- `src/jspace/`: the shared code. `model.Session` loads a model and its J-lens at pinned revisions and runs batched forward passes and generation with hooks; `lens` reads ranks and builds subspaces and directions; `hooks` builds the interventions; `stimuli` draws the pictures; `judge` scores answers on natural photos; `analysis` holds scoring and the bootstrap; `runs` makes run folders; `config` pins revisions and paths.
- `experiments/`: one script per experiment. Each one parses its arguments, opens a `runs.Run`, loads a `Session`, and writes its CSV of every answer, `summary.json`, `stimuli.png` and `config.json` to `runs/<run>/<model>/`. The docstring states the design, the controls and the outputs.
- `scripts/`: `download.py` (data, models and lenses, checked against the original runs), `reproduce.sh`, `make_figures.py`, `check.py`.
- `results/`: the outputs of the original runs, the reference for `check.py`. Never edit them.
- `tests/`: CPU tests, no model and no network.

## Commands

```bash
uv sync                                                   # the environment, exactly as uv.lock pins it
uv run pytest -q                                          # the tests
uv run ruff format src experiments scripts tests && uv run ruff check src experiments scripts tests
uv run experiments/<name>.py --n 4 --run try_<name>       # a small GPU run (--help lists the arguments)
uv run scripts/check.py                                   # compare runs/ with results/
uv run scripts/make_figures.py --runs results             # the figures from the original outputs
```

## Scientific invariants

`results/` was produced with the following. Changing any of them makes a new experiment, which needs its own run name and its own rule written before it runs; it is never a fix to an existing one.

- Prompts, question wordings, the answer scoring, and the judge's model and rubrics (the rubric text is hashed into the verdict cache key).
- Seeds and the order of random draws: the crop shuffle, scene sampling, random subspaces and directions, and `torch.manual_seed(0)` right after the model loads.
- The crop sets and the crop gate, the canvas geometry, the DejaVu fonts, the photo resizing.
- Read layers, ranks (r = 0.1 d), doses, window size, the calibration split.
- The rules that set each verdict in `summary.json`: their thresholds were written before the runs.
- The pinned revisions in `src/jspace/config.py` and the package versions in `uv.lock`.
- Batching: batch sizes follow GPU memory, and a different batch composition changes floating-point results.

## Rules for changes

- A change to `src/jspace/` or `experiments/` must leave the outputs identical. Run each affected experiment at small size (`--n 4`) before and after the change, compare the two run folders (`uv run scripts/check.py --runs <after> --reference <before>`), and run the tests.
- A new experiment is a new script in `experiments/` that uses the package. It gets a docstring with the design, the controls, the outputs and the rule written before the run.
- Outputs go through `runs.Run` into `runs/`. A missing input stops a run with an error: no silent defaults (an answer without a judge verdict, a missing horizons file).
- Inspect the answers themselves (the CSVs, `stimuli.png`), not only the summaries.
- Secrets live in `.env`, which git ignores; never print or commit them.
- Add a dependency only with a reason, pinned in `pyproject.toml` and locked with `uv lock`.
- Style is ruff's (`pyproject.toml`); comments say why the code does what it does.
