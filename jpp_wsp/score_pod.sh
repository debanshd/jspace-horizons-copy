#!/usr/bin/env bash
# On a GPU pod: score the weight-space-programmed lenses (jpp_wsp/RULE.md) beside the 50% fit's own J++ lens and the logit lens,
# with the J++ post's evals, then push the outputs to this repository's branch.
#
# Env: GITHUB_TOKEN (the pod's secret), BRANCH (where the outputs go), WSP_URLS (base64 JSON {file name: URL}: the four variants
# and the target lens, as short-lived signed links, since the pod holds no Hugging Face token). The 50% fit's two correctness
# files come from jpp_wsp/inputs/ on the branch, committed after the edit was frozen. Every lens is checked against
# jpp_wsp/frozen.json before anything runs.
set -euo pipefail

WORK=/workspace/wsp
mkdir -p "$WORK"
exec > >(tee -a "$WORK/pod.log") 2>&1
export HF_HOME=$WORK/hf
REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/debanshd/jspace-horizons-copy.git"

push_outputs() {
  # the outputs and the log, whatever happened; the lens files stay on the pod
  cd "$WORK/repo"
  mkdir -p jpp_wsp/out
  if [ -d "$WORK/out" ]; then cp -r "$WORK/out/." jpp_wsp/out/ && rm -rf jpp_wsp/out/lenses; fi
  cp "$WORK/pod.log" jpp_wsp/out/pod.log
  git add jpp_wsp/out
  git -c user.name=wsp-pod -c user.email=wsp-pod@users.noreply.github.com commit -q -m "jpp_wsp: scores from the pod ($1)" || true
  for i in 1 2 3 4; do
    git fetch -q origin "$BRANCH" && git merge -q --no-edit FETCH_HEAD && git push -q origin "HEAD:$BRANCH" && return 0
    sleep $((i * 5))
  done
}
trap 'push_outputs failed' ERR

git clone -q "$REPO_URL" "$WORK/repo"
git -C "$WORK/repo" checkout -q "$BRANCH"
# PR 18's eval driver at the commit the 50% fit was scored with, beside (not in) the branch's tree
git -C "$WORK/repo" fetch -q origin pr-b-olmo-jpp
git -C "$WORK/repo" worktree add -q "$WORK/pr18" 3d8ccd6

FITS=$WORK/fits/olmo-3-7b-50pct
mkdir -p "$FITS"
python3 - "$FITS" "$WORK/repo/jpp_wsp/frozen.json" <<'EOF'
import base64, hashlib, json, os, subprocess, sys
fits, frozen = sys.argv[1], json.load(open(sys.argv[2]))
urls = json.loads(base64.b64decode(os.environ["WSP_URLS"]))
for name, url in urls.items():
    out = os.path.join(fits, name)
    subprocess.run(["curl", "-sSfL", "--retry", "4", "-o", out, url], check=True)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    want = frozen["sha256"].get(name)
    if want is None or h.hexdigest() != want:
        raise SystemExit(f"{name}: sha256 {h.hexdigest()}, frozen.json has {want}")
    print(name, "ok", flush=True)
EOF
# the split the target was scored on: run_all.py --fits reads it from the fit's folder rather than regrading
cp "$WORK/repo/jpp_wsp/inputs/model_correctness.csv" "$WORK/repo/jpp_wsp/inputs/correctness.csv" "$FITS/"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -q https://github.com/safety-research/jpp_lens.git "$WORK/jpp_lens"
cd "$WORK/jpp_lens"
git checkout -q d7efc0de372e20589e899188c0cfe8434acc63ec
uv sync -q --extra cu128
uv run python "$WORK/pr18/jpp_evals/run_all.py" --out "$WORK/out" --fits "$WORK/fits" olmo-3-7b-50pct

trap - ERR
push_outputs done
echo "WSP_DONE"
sleep infinity
