#!/usr/bin/env bash
# On a GPU pod, driven through Jupyter by jpp_wsp/pod.py: score experiments 1 and 2 (RULE.md, RULE2.md) in one run, beside the
# 50% fit's own J++ lens and the logit lens, with the J++ post's evals. Replaces score_pod.sh, whose pod stalled at start-up before
# logging anything: this one needs no GitHub token (pod.py uploads the code and inputs and reads the outputs back) and logs each stage.
#
# Expects in $WORK (/workspace/wsp): code/jpp_evals (PR 18 at 3d8ccd6), code/frozen.json and code/frozen2.json, inputs/*.csv (the
# 50% fit's correctness files), urls.json ({file name: signed URL}). Writes out/ and, at the end, DONE or FAILED.
set -euo pipefail
WORK=/workspace/wsp
cd "$WORK"
trap 'echo "FAILED at line $LINENO"; touch "$WORK/FAILED"' ERR
stage() { echo "=== $(date -u +%H:%M:%S) $*"; }
export HF_HOME=$WORK/hf

stage "download and check the lenses"
FITS=$WORK/fits/olmo-3-7b-50pct
mkdir -p "$FITS"
python3 - "$FITS" <<'EOF'
import hashlib, json, subprocess, sys
fits = sys.argv[1]
want = {}
for f in ("code/frozen.json", "code/frozen2.json"):
    want |= json.load(open(f))["sha256"]
for name, url in json.load(open("urls.json")).items():
    out = f"{fits}/{name}"
    subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", out, url], check=True)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != want.get(name):
        raise SystemExit(f"{name}: sha256 {h.hexdigest()}, frozen has {want.get(name)}")
    print(name, "ok", flush=True)
EOF
# the split the target was scored on: run_all.py --fits reads it from the fit's folder rather than regrading
cp inputs/model_correctness.csv inputs/correctness.csv "$FITS/"

stage "the harness at d7efc0d"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d jpp_lens/.git ] || git clone -q https://github.com/safety-research/jpp_lens.git jpp_lens
cd jpp_lens
git checkout -q d7efc0de372e20589e899188c0cfe8434acc63ec
uv sync -q --extra cu128

stage "the evals"
uv run python "$WORK/code/jpp_evals/run_all.py" --out "$WORK/out" --fits "$WORK/fits" olmo-3-7b-50pct
stage "done"
touch "$WORK/DONE"
