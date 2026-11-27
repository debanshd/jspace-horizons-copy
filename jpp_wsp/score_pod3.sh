#!/usr/bin/env bash
# On a GPU pod, driven through Jupyter by jpp_wsp/pod.py: experiment 3 (RULE3.md), jeeva2812's OLMo 3 7B J-lens and the logit lens
# on the 100% model, on the held-out split the 100% lenses were scored on before. The harness's run_cli.py is called directly, so
# only these two are scored. Expects in $WORK: code/jpp_evals, code/frozen3.json, code/convert_jall.py, inputs/*.csv (the 100% fit's
# correctness files), urls.json. Writes out/olmo-3-7b/ and DONE or FAILED.
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
OUT=$WORK/out/olmo-3-7b
mkdir -p "$OUT"
CLI=("$WORK/code/jpp_evals/run_cli.py")
COMMON=(--model-key olmo-3-7b --hf-model-name allenai/Olmo-3-1025-7B --layers 4,8,12,16,20,24,28)
LENSES=(--lens "$FITS/extra/jeeva_Jall_main.pt" logit)
ITEMS=(--items held-out:0 --correctness-csv "$WORK/inputs/model_correctness.csv")
uv run python "${CLI[@]}" evaluate "${COMMON[@]}" "${LENSES[@]}" "${ITEMS[@]}" --out "$OUT/readout_filtered.csv"
stage "readout_filtered done"
uv run python "${CLI[@]}" evaluate "${COMMON[@]}" "${LENSES[@]}" "${ITEMS[@]}" --no-readout-filter --out "$OUT/readout_full.csv"
stage "readout_full done"
uv run python "${CLI[@]}" probe-swap "${COMMON[@]}" "${LENSES[@]}" --out "$OUT/probe_swap.parquet"
stage "probe_swap done"
stage "done"
touch "$WORK/DONE"
