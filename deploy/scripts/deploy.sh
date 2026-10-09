#!/usr/bin/env bash
# Update the server to the newest commit and reload the PM2 processes.
#   bash /opt/speechjudge/repo/speech-judge/deploy/scripts/deploy.sh [branch]
# Run as the login user that owns PM2 (ubuntu). Environment:
#   PM2_APPS   which processes to run: "judge-api" (Layout A, default) or "judge-api,judge-worker" (Layout B)
set -euo pipefail

BRANCH="${1:-main}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"          # .../speech-judge
REPO_DIR="$(git -C "$APP_DIR" rev-parse --show-toplevel)"
APPS="${PM2_APPS:-judge-api}"

cd "$REPO_DIR"
before="$(cat "$APP_DIR"/requirements*.txt | sha256sum)"
# benchmarks/benchmark_results.json is tracked (as an empty template) but filled in by the benchmark
# scripts on this server: keep the measured numbers across the hard reset below.
RESULTS="$APP_DIR/benchmarks/benchmark_results.json"
[ -f "$RESULTS" ] && cp "$RESULTS" "$RESULTS.keep"
git fetch --prune origin
git checkout -q "$BRANCH"
git reset --hard "origin/$BRANCH"                                       # .env and .venv are untracked, so they are kept
[ -f "$RESULTS.keep" ] && mv "$RESULTS.keep" "$RESULTS"
after="$(cat "$APP_DIR"/requirements*.txt | sha256sum)"

cd "$APP_DIR"
if [ "$before" != "$after" ]; then
  echo "requirements changed: installing"
  .venv/bin/pip install -r requirements.txt -r requirements-ml.txt
fi

echo "deploying $(git -C "$REPO_DIR" rev-parse --short HEAD) with: $APPS"
pm2 startOrReload deploy/pm2/ecosystem.config.js --only "$APPS" --update-env
pm2 save
bash deploy/scripts/wait_ready.sh
