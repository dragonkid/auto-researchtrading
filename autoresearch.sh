#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?Usage: ./autoresearch.sh <tag> [max_experiments] [council_threshold]}"
BRANCH="autotrader/${TAG}"
RESULTS="results.tsv"
MAX_EXPERIMENTS="${2:-0}"
COUNCIL_THRESHOLD="${3:-5}"
EXPERIMENT_COUNT=0
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
echo "Max experiments: ${MAX_EXPERIMENTS:-unlimited}"
echo "Council threshold: $COUNCIL_THRESHOLD (discards or plateau)"
echo ""

# Count consecutive discards from the tail of results.tsv
count_consecutive_discards() {
  if [ ! -f "$RESULTS" ]; then
    echo 0
    return
  fi
  tail -n +2 "$RESULTS" | awk -F'\t' '{print $5}' | tail -r | awk '
    /^discard$/ { count++; next }
    { exit }
    END { print count+0 }
  '
}

# Detect score plateau: recent N experiments improved less than median keep delta
# Returns 1 if plateau detected, 0 otherwise
check_score_plateau() {
  local window="$1"
  if [ ! -f "$RESULTS" ]; then
    echo 0
    return
  fi

  tail -n +2 "$RESULTS" | awk -F'\t' -v window="$window" '
  {
    scores[NR] = $2 + 0
    statuses[NR] = $5
    n = NR
  }
  END {
    if (n < window) { print 0; exit }

    # Collect all keep scores in order
    keep_count = 0
    for (i = 1; i <= n; i++) {
      if (statuses[i] == "keep") {
        keep_count++
        keep_scores[keep_count] = scores[i]
      }
    }
    if (keep_count < 3) { print 0; exit }

    # Compute keep deltas (improvement per keep)
    delta_count = 0
    for (i = 2; i <= keep_count; i++) {
      delta_count++
      deltas[delta_count] = keep_scores[i] - keep_scores[i-1]
    }
    if (delta_count < 2) { print 0; exit }

    # Sort deltas to find median (simple insertion sort)
    for (i = 2; i <= delta_count; i++) {
      key = deltas[i]
      j = i - 1
      while (j > 0 && deltas[j] > key) {
        deltas[j+1] = deltas[j]
        j--
      }
      deltas[j+1] = key
    }
    if (delta_count % 2 == 1)
      median = deltas[int(delta_count/2) + 1]
    else
      median = (deltas[delta_count/2] + deltas[delta_count/2 + 1]) / 2

    # Best keep score before the window
    best_before = -999
    cutoff = n - window
    for (i = 1; i <= cutoff; i++) {
      if (statuses[i] == "keep" && scores[i] > best_before) {
        best_before = scores[i]
      }
    }
    if (best_before <= -999) { print 0; exit }

    # Best keep score within the window
    best_in_window = -999
    for (i = cutoff + 1; i <= n; i++) {
      if (statuses[i] == "keep" && scores[i] > best_in_window) {
        best_in_window = scores[i]
      }
    }
    if (best_in_window <= -999) best_in_window = best_before

    recent_improvement = best_in_window - best_before
    plateau = (recent_improvement < median) ? 1 : 0
    print plateau
  }'
}

# Run a Council Mode session
run_council() {
  local reason="$1"
  local council_num=$((COUNCIL_COUNT + 1))
  echo ""
  echo "========================================================"
  echo "  COUNCIL MODE #${council_num} — ${reason}"
  echo "========================================================"
  echo ""

  local council_output
  council_output=$(CLAUDE_CONFIG_DIR=~/.claude-autoresearch claude -p \
    --dangerously-skip-permissions \
    --model opus \
    --effort max \
    --system-prompt-file "$PROJECT_DIR/program-council.md" \
    --allowedTools "Read,Edit,Write,Bash(git:*),Bash(uv run:*),Bash(grep:*),Bash(tail:*),Bash(head:*),Bash(cat:*),Bash(echo:*),Grep,Glob" \
    "Run Council Mode. Read program-council.md for instructions. This is Council Session #${council_num}. Trigger reason: ${reason}." \
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
  # Check convergence: consecutive discards OR score plateau
  consecutive_discards=$(count_consecutive_discards)
  plateau=$(check_score_plateau "$COUNCIL_THRESHOLD")

  if [ "$consecutive_discards" -ge "$COUNCIL_THRESHOLD" ]; then
    if ! run_council "${COUNCIL_THRESHOLD} consecutive discards"; then
      break
    fi
  elif [ "$plateau" -eq 1 ]; then
    if ! run_council "score plateau (recent improvement < median keep delta)"; then
      break
    fi
  fi

  EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
  if [ "$MAX_EXPERIMENTS" -gt 0 ] && [ "$EXPERIMENT_COUNT" -gt "$MAX_EXPERIMENTS" ]; then
    echo "Reached max experiments: $MAX_EXPERIMENTS"
    break
  fi

  # Clean up any leftover state from interrupted experiments
  git checkout -- strategy.py 2>/dev/null || true

  echo "=== Experiment $EXPERIMENT_COUNT ($(date '+%H:%M:%S')) ==="

  CLAUDE_CONFIG_DIR=~/.claude-autoresearch claude -p \
    --dangerously-skip-permissions \
    --effort max \
    --system-prompt-file "$PROJECT_DIR/program-stateless.md" \
    --allowedTools "Read" "Edit" "Write" "Bash(git:*)" "Bash(uv run:*)" "Bash(grep:*)" "Bash(tail:*)" "Bash(head:*)" "Bash(cat:*)" "Grep" "Glob" \
    -- \
    "Run one experiment. Read program-stateless.md and results.tsv for context. Run 'git log main..HEAD --oneline' for this branch's experiment history only. Modify strategy.py, commit, backtest, record result, then EXIT." \
    || {
      echo "Claude exited with error (code $?), continuing after cooldown..."
      sleep 5
    }

  # Ensure results.tsv ends with a newline (agent sometimes uses Write/Edit tool which strips it)
  [ -f "$RESULTS" ] && [ -n "$(tail -c1 "$RESULTS")" ] && echo >> "$RESULTS"

  echo ""
done
