#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?Usage: ./autoresearch.sh <tag> [max_experiments]}"
BRANCH="autotrader/${TAG}"
RESULTS="results.tsv"
MAX_EXPERIMENTS="${2:-0}"
EXPERIMENT_COUNT=0

cd "$PROJECT_DIR"

# Initialize: create branch if it doesn't exist
if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH"
  echo -e "commit\tscore\tsharpe\tmax_dd\tstatus\tdescription" > "$RESULTS"
  git add "$RESULTS"
  git commit -m "init: clean slate for ${TAG}"
else
  git checkout "$BRANCH"
fi

echo "Branch: $BRANCH"
echo "Max experiments: ${MAX_EXPERIMENTS:-unlimited}"
echo ""

# Main loop
while true; do
  EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
  if [ "$MAX_EXPERIMENTS" -gt 0 ] && [ "$EXPERIMENT_COUNT" -gt "$MAX_EXPERIMENTS" ]; then
    echo "Reached max experiments: $MAX_EXPERIMENTS"
    break
  fi

  echo "=== Experiment $EXPERIMENT_COUNT ($(date '+%H:%M:%S')) ==="

  CLAUDE_CONFIG_DIR=~/.claude-autoresearch claude -p \
    --dangerously-skip-permissions \
    --model opus \
    --effort max \
    --system-prompt-file "$PROJECT_DIR/program-stateless.md" \
    --allowedTools "Read,Edit,Write,Bash(git:*),Bash(uv run:*),Bash(grep:*),Bash(tail:*),Bash(head:*),Bash(cat:*),Grep,Glob" \
    "Run one experiment. Read program-stateless.md and results.tsv for context. Run 'git log main..HEAD --oneline' for this branch's experiment history only. Modify strategy.py, commit, backtest, record result, then EXIT." \
    || {
      echo "Claude exited with error (code $?), continuing after cooldown..."
      sleep 5
    }

  echo ""
done
