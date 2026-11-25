#!/usr/bin/env bash
# The Qwen rule's calibration (config.QWEN_CALIBRATION, scripts/check_lens.py's docstring): check_lens --ranks on the four
# Neuronpedia Qwen lenses the original results read, Qwen3.5-2B, 4B and 27B on their J-lenses and Qwen3.5-9B on its J++ lens
# (its pinned lens), into runs/check_lens_calibration_ranks. Their ranks set the Qwen rule's bar, mechanically, before any lens
# is judged by it. At the end it prints whether the bar rejects each model's own logit lens: if not for all four, the rule is
# void and no Qwen lens passes it (a bar a lens with no Jacobian can meet says nothing).
#
# History: the first calibration run (runs/check_lens_calibration, 25 Nov) measured top-token agreement only, and found the first
# rule fails three of the four. It stays as it is.
#
#   scripts/lens_calibration.sh            one H200 (the 27B), about 15 minutes with the downloads
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=${MODELS:-"Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B"}
RUN=check_lens_calibration_ranks

uv sync --frozen || exit 1
for m in $MODELS; do
    tag=${m#*/}
    echo "===== $m"
    uv run --frozen scripts/download.py --skip-data --models "$m" || { echo "$m: download failed" >&2; continue; }
    [ -f "runs/$RUN/$tag/summary.json" ] || uv run --frozen scripts/check_lens.py --model "$m" --run "$RUN" --ranks ||
        echo "$m: check_lens did not run to the end: see runs/$RUN/$tag/run.log" >&2
done
uv run --frozen python - "runs" "$RUN" <<'PY'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("check_lens", "scripts/check_lens.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
cal = cl.load_calibration(Path(sys.argv[1]), sys.argv[2])
for m, s in cal.items():
    print(m, "J-lens log10 rank", s["rank_jlens"], "logit lens", s["rank_logit"])
inf = cl.informative(cal)
print("the bar rejects each model's logit lens:", inf, "->", "the Qwen rule is usable" if all(inf.values()) else "the Qwen rule is VOID: no Qwen lens passes it")
PY
