#!/usr/bin/env bash
# On a GPU pod, driven through Jupyter by jpp_wsp/pod.py: experiment 3 (RULE3.md), jeeva2812's OLMo 3 7B J-lens beside the logit
# lens, Neuronpedia's J-lens and the 100% J++ lens, on the 100% model. Expects in $WORK: code/jpp_evals, code/frozen3.json,
# code/convert_jall.py, code/score_extra.py, inputs/*.csv (the 100% fit's correctness files), urls.json. Writes out/ and DONE or FAILED.
set -euo pipefail
WORK=/workspace/wsp
cd "$WORK"
trap 'echo "FAILED at line $LINENO"; touch "$WORK/FAILED"' ERR
stage() { echo "=== $(date -u +%H:%M:%S) $*"; }
export HF_HOME=$WORK/hf

stage "download and check the lenses"
FITS=$WORK/fits/olmo-3-7b
mkdir -p "$FITS/extra" "$WORK/src"
python3 - <<'EOF'
import hashlib, json, subprocess
want = json.load(open("code/frozen3.json"))["sha256"]
for name, url in json.load(open("urls.json")).items():
    out = f"src/{name}"
    subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", out, url], check=True)
    h = hashlib.sha256()
    with open(out, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != want.get(name):
        raise SystemExit(f"{name}: sha256 {h.hexdigest()}, frozen3.json has {want.get(name)}")
    print(name, "ok", flush=True)
EOF
mv src/olmo7b_jpp_n64.pt "$FITS/"
cp inputs/model_correctness.csv inputs/correctness.csv "$FITS/"

stage "the harness at d7efc0d"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d jpp_lens/.git ] || git clone -q https://github.com/safety-research/jpp_lens.git jpp_lens
cd jpp_lens
git checkout -q d7efc0de372e20589e899188c0cfe8434acc63ec
uv sync -q --extra cu128

stage "convert the jeeva lens"
uv run python "$WORK/code/convert_jall.py" "$WORK/src/Jall_main.pt" "$FITS/extra/jeeva_Jall_main.pt"

stage "the evals"
uv run python "$WORK/code/score_extra.py" "$WORK/code/jpp_evals" --out "$WORK/out" --fits "$WORK/fits" olmo-3-7b
stage "done"
touch "$WORK/DONE"
