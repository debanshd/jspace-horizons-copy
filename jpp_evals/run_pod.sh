#!/usr/bin/env bash
# On a GPU pod: fetch the J++ harness at its pinned commit, build its environment, run the evals, pack the outputs.
#
#   jpp_evals/run_pod.sh                      # the default models (manifest.py keys)
#   jpp_evals/run_pod.sh qwen2.5-omni-3b      # some of them, by key (manifest.py all-keys lists them)
#
# Run from this repository's root. WORK (default /workspace/jpp) holds the checkout, the Hugging Face cache and the outputs;
# PRUNE=0 keeps every model in the cache (by default each is deleted once its evals are done, to fit a 200 GB volume).
# Every file it downloads is public, so no token is needed. The harness's environment pins torch for CUDA 12.8: pick a pod
# image whose driver supports it.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-/workspace/jpp}
export HF_HOME=${HF_HOME:-$WORK/hf}
mkdir -p "$WORK"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# The harness keeps its own uv.lock (torch 2.11, transformers 5.14.1), apart from this repository's, which stays as it is.
COMMIT=$(python3 "$HERE/manifest.py" jpp-commit)
if [ ! -d "$WORK/jpp_lens/.git" ]; then
  git clone --quiet https://github.com/safety-research/jpp_lens.git "$WORK/jpp_lens"
fi
git -C "$WORK/jpp_lens" checkout --quiet "$COMMIT"
cd "$WORK/jpp_lens"
uv sync --extra cu128

PRUNE_FLAG=()
if [ "${PRUNE:-1}" = 1 ]; then PRUNE_FLAG=(--prune); fi

if [ "${FIT:-0}" = 1 ]; then
  # FIT=1: fit an R-lens and a J++ lens for Qwen2.5-Omni-7B's thinker (FIT_PROMPTS, default 64, the post's and Neuronpedia's count), then
  # score them beside the J-lenses, every lens of the model on the held-out split. Outputs in $WORK/fits and $WORK/out_fits.
  KEYS=("$@")
  if [ ${#KEYS[@]} -eq 0 ]; then KEYS=(qwen2.5-omni-7b); fi
  # CONVERGE=1 instead fits blocks of 25 prompts until Neuronpedia's convergence rule stops it, at most 1000 (fit_lenses.py --converge).
  FIT_ARGS=(--num-prompts "${FIT_PROMPTS:-64}")
  if [ "${CONVERGE:-0}" = 1 ]; then FIT_ARGS=(--converge); fi
  uv run python "$HERE/fit_lenses.py" --fits "$WORK/fits" "${FIT_ARGS[@]}" "${KEYS[@]}" 2>&1 | tee -a "$WORK/fit_lenses.log"
  uv run python "$HERE/run_all.py" --out "$WORK/out_fits" --fits "$WORK/fits" "${PRUNE_FLAG[@]}" "${KEYS[@]}" 2>&1 | tee -a "$WORK/run_all.log"
  # The tables, logs and expert weights; the lens files (1.4 GB each) stay in $WORK/fits.
  tar czf "$WORK/jpp_fits_out.tgz" --exclude out_fits/lenses --exclude 'fits/*/artifacts' --exclude 'fits/*/*.pt' \
    -C "$WORK" out_fits fits fit_lenses.log run_all.log
  echo "lenses: $WORK/fits/<model>/*_rlens_n*.pt and *_jpp_n*.pt; scores: $WORK/out_fits (packed in $WORK/jpp_fits_out.tgz)"
  exit 0
fi

uv run python "$HERE/run_all.py" --out "$WORK/out" "${PRUNE_FLAG[@]}" "$@" 2>&1 | tee -a "$WORK/run_all.log"

# Everything but the lens files (links into the cache; their commits and sha256 are in manifest.py).
tar czf "$WORK/jpp_evals_out.tgz" --exclude out/lenses -C "$WORK" out run_all.log
echo "outputs: $WORK/out (packed in $WORK/jpp_evals_out.tgz)"
