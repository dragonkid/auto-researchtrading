#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?Usage: ./autoresearch.sh <tag> [max_rounds] [council_threshold]}"
BRANCH="autotrader/${TAG}"
RESULTS="results.tsv"
MAX_ROUNDS="${2:-0}"
COUNCIL_THRESHOLD="${3:-5}"
ROUND_COUNT=0
COUNCIL_COUNT=0

cd "$PROJECT_DIR"

# Ensure Ctrl+C stops the entire script
trap 'echo ""; echo "Interrupted. Cleaning up..."; git checkout -- strategy.py 2>/dev/null; exit 130' INT TERM

# Initialize: create branch if it doesn't exist
if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -b "$BRANCH"
else
  git checkout "$BRANCH"
fi

# Ensure results.tsv exists with header (untracked, not committed)
if [ ! -f "$RESULTS" ]; then
  echo -e "commit\tscore\tmean_score\tstd_score\tstatus\tdescription" > "$RESULTS"
fi

echo "Branch: $BRANCH"
echo "Max rounds: ${MAX_ROUNDS:-unlimited} (each round = up to 20 experiments, exits on 5 consecutive discards)"
echo "Council threshold: $COUNCIL_THRESHOLD consecutive discards"
echo ""

# Count consecutive discards from the tail of results.tsv
count_consecutive_discards() {
  if [ ! -f "$RESULTS" ]; then
    echo 0
    return
  fi
  # Read status column (5th field), count consecutive "discard" from bottom
  tail -n +2 "$RESULTS" | awk -F'\t' '{print $5}' | tail -r | awk '
    /^discard$/ { count++; next }
    { exit }
    END { print count+0 }
  '
}

# Run a Council Mode session
run_council() {
  local council_num=$((COUNCIL_COUNT + 1))
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  COUNCIL MODE #${council_num} — ${COUNCIL_THRESHOLD} consecutive discards detected  ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""

  local council_output
  council_output=$(CLAUDE_CONFIG_DIR=~/.claude-autoresearch codemax claude -p \
    --dangerously-skip-permissions \
    --model opus \
    --effort max \
    --system-prompt-file "$PROJECT_DIR/program-council.md" \
    --allowedTools "Read,Edit,Write,Bash(git:*),Bash(uv run:*),Bash(grep:*),Bash(tail:*),Bash(head:*),Bash(cat:*),Bash(echo:*),Grep,Glob" \
    "Run Council Mode. Read program-council.md for instructions. This is Council Session #${council_num}." \
    2>&1) || true

  COUNCIL_COUNT=$council_num

  # Parse verdict from output
  if echo "$council_output" | grep -q "COUNCIL_VERDICT: ACCEPT"; then
    echo ""
    echo "Council #${council_num}: ACCEPT — breakthrough found, resuming experiments"
    echo ""
    return 0  # continue looping
  elif echo "$council_output" | grep -q "COUNCIL_VERDICT: PASS"; then
    echo ""
    echo "Council #${council_num}: PASS — strategy confirmed near-optimal"
    echo "Auto-stopping autoresearch."
    echo ""
    return 1  # signal to stop
  else
    echo ""
    echo "Council #${council_num}: no clear verdict detected, treating as PASS"
    echo "Auto-stopping autoresearch."
    echo ""
    return 1  # fail-safe: stop if verdict is unclear
  fi
}

# Main loop
while true; do
  # Check convergence before each experiment
  consecutive_discards=$(count_consecutive_discards)
  if [ "$consecutive_discards" -ge "$COUNCIL_THRESHOLD" ]; then
    if ! run_council; then
      break
    fi
    # Council ACCEPT: reset and continue
  fi

  ROUND_COUNT=$((ROUND_COUNT + 1))
  if [ "$MAX_ROUNDS" -gt 0 ] && [ "$ROUND_COUNT" -gt "$MAX_ROUNDS" ]; then
    echo "Reached max rounds: $MAX_ROUNDS"
    break
  fi

  # Clean up any leftover state from interrupted experiments
  git checkout -- strategy.py 2>/dev/null || true

  echo "=== Round $ROUND_COUNT ($(date '+%H:%M:%S')) ==="

  MAX_THINKING_TOKENS=256000 CLAUDE_CODE_EFFORT_LEVEL=max \
  CLAUDE_CONFIG_DIR=~/.claude-autoresearch codemax claude -p \
    --dangerously-skip-permissions \
    --effort max \
    --system-prompt-file "$PROJECT_DIR/program-stateless.md" \
    --allowedTools "Read" "Edit" "Write" "Bash(git:*)" "Bash(uv run:*)" "Bash(grep:*)" "Bash(tail:*)" "Bash(head:*)" "Bash(cat:*)" "Grep" "Glob" \
    -- \
    "Working directory: $PROJECT_DIR. Run a research session. FIRST: Read program-stateless.md — it contains MANDATORY diagnostic steps and the EXIT RULE you must follow. Then read results.tsv (from 3rd-last keep onward) and run 'git log main..HEAD --oneline' for context. For each experiment: modify strategy.py, commit, backtest, record to results.tsv, then decide whether to continue or exit. Follow the EXIT RULE in program-stateless.md exactly — you CANNOT exit without having attempted at least 2 architectural changes among your discards." \
    || {
      echo "Claude exited with error (code $?), continuing after cooldown..."
      sleep 5
    }

  # Ensure results.tsv ends with a newline (agent sometimes uses Write/Edit tool which strips it)
  [ -f "$RESULTS" ] && [ -n "$(tail -c1 "$RESULTS")" ] && echo >> "$RESULTS"

  echo ""
done
