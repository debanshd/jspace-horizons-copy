#!/bin/bash
# Push main to GitHub with a token read from /workspace/nayan/.env (GITHUB_TOKEN=...), never printed.
# The token needs the "repo" scope (classic) or contents: write (fine-grained) on the-infiltrator/jspace-horizons.
set -e
cd /workspace/nayan/jspace-horizons
set -a; . /workspace/nayan/.env; set +a
[ -n "$GITHUB_TOKEN" ] || { echo "GITHUB_TOKEN is not set in /workspace/nayan/.env"; exit 1; }
git -c credential.helper='!f() { echo "username=x-access-token"; echo "password=$GITHUB_TOKEN"; }; f' push origin main 2>&1 | sed "s/$GITHUB_TOKEN/[token]/g"
echo "pushed $(git rev-parse --short HEAD) to origin/main"
