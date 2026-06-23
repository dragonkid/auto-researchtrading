#!/usr/bin/env bash
set -uo pipefail
# NOTE: removed 'set -e' — it caused silent exits when subcommands
# (claude, git, awk pipelines) returned non-zero unexpectedly.
# Errors are handled explicitly via || clauses.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?Usage: ./autoresearch.sh <tag> [max_rounds] [council_threshold] [provider] [model]}"
BRANCH="autotrader/${TAG}"
RESULTS="results.tsv"
MAX_ROUNDS="${2:-0}"
COUNCIL_THRESHOLD="${3:-5}"
# Provider/model are optional. When PROVIDER is set, credentials (base_url +
# api_key + default model) are resolved at runtime from the Hermes profile
# config via scripts/hermes_provider.py — keys are NEVER stored in this
# (git-tracked) script. MODEL overrides the provider's default model.
#   provider names live under custom_providers: in
#   ~/.hermes/profiles/quant-trading/config.yaml  (currently: skyapi, litellm, litellmqa)
PROVIDER="${4:-skyapi}"
# MODEL: explicit 5th arg wins; else the provider's configured default model.
MODEL_OVERRIDE="${5:-}"
ROUND_COUNT=0
COUNCIL_COUNT=0

cd "$PROJECT_DIR"

# ── Resolve provider credentials from Hermes config (no secrets in this file) ──
HERMES_CONFIG="${HERMES_CONFIG:-$HOME/.hermes/profiles/quant-trading/config.yaml}"
_resolved=$(python3 "$PROJECT_DIR/scripts/hermes_provider.py" "$PROVIDER" --config "$HERMES_CONFIG") || {
  echo "ERROR: could not resolve provider '$PROVIDER' from $HERMES_CONFIG" >&2
  exit 1
}
PROVIDER_BASE_URL=$(printf '%s' "$_resolved" | cut -f1)
PROVIDER_API_KEY=$(printf '%s' "$_resolved" | cut -f2)
PROVIDER_MODEL=$(printf '%s' "$_resolved" | cut -f3)
unset _resolved
# Anthropic SDK appends /v1/messages itself; strip a trailing /v1 from base_url.
PROVIDER_BASE_URL="${PROVIDER_BASE_URL%/v1}"
# CLI model arg: explicit override wins, else the provider's configured default.
MODEL="${MODEL_OVERRIDE:-$PROVIDER_MODEL}"
if [ -z "$MODEL" ]; then
  echo "ERROR: no model resolved for provider '$PROVIDER' and none given" >&2
  exit 1
fi
if [ -z "$PROVIDER_BASE_URL" ] || [ -z "$PROVIDER_API_KEY" ]; then
  echo "ERROR: provider '$PROVIDER' missing base_url or api_key in $HERMES_CONFIG" >&2
  exit 1
fi

# ── Reasoning-mode configuration ──
# Claude Code's Anthropic SDK emits `{"type":"thinking", ...}` content blocks
# when extended thinking is enabled. Non-Anthropic OpenAI-compatible endpoints
# (sglang/vLLM/litellm-proxy with GLM, Qwen, etc.) reject `thinking` blocks
# with HTTP 400 because their OpenAI schema only accepts text/image/audio/
# tool_reference parts. Native reasoning models (GLM-5.2, etc.) do their own
# chain-of-thought via reasoning_content and don't need Claude's thinking
# blocks — so we disable extended thinking for non-Anthropic models.
#
# Detection: model name pattern. Anthropic-native models (claude-*) keep
# thinking enabled; everything else (GLM, Qwen, DeepSeek, etc.) disables it.
case "$MODEL" in
  claude-*)
    THINKING_BUDGET=64000
    EFFORT="max"
    ;;
  *)
    THINKING_BUDGET=0
    EFFORT="high"
    ;;
esac

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
echo "Provider: $PROVIDER  (base_url: $PROVIDER_BASE_URL)"
echo "Model: $MODEL"
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
  tail -n +2 "$RESULTS" | awk -F'\t' '{print $9}' | tail -r | awk '
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
  council_output=$(ANTHROPIC_BASE_URL="$PROVIDER_BASE_URL" \
    ANTHROPIC_AUTH_TOKEN="$PROVIDER_API_KEY" \
    ANTHROPIC_MODEL="$MODEL" \
    CLAUDE_CONFIG_DIR=~/.claude-autoresearch claude -p \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --effort "$EFFORT" \
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

ANTHROPIC_BASE_URL="$PROVIDER_BASE_URL" \
  ANTHROPIC_AUTH_TOKEN="$PROVIDER_API_KEY" \
  ANTHROPIC_MODEL="$MODEL" \
  MAX_THINKING_TOKENS="$THINKING_BUDGET" CLAUDE_CODE_EFFORT_LEVEL="$EFFORT" \
  CLAUDE_CONFIG_DIR=~/.claude-autoresearch claude -p \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --effort "$EFFORT" \
    --system-prompt-file "$PROJECT_DIR/program-stateless.md" \
    --allowedTools "Read" "Edit" "Write" "Bash(git:*)" "Bash(uv run:*)" "Bash(grep:*)" "Bash(tail:*)" "Bash(head:*)" "Bash(cat:*)" "Grep" "Glob" \
    -- \
    "Working directory: $PROJECT_DIR. Run a research session. FIRST: Read program-stateless.md — it contains MANDATORY diagnostic steps and the EXIT RULE you must follow. Then read results.tsv (from 3rd-last keep onward) and run 'git log main..HEAD --oneline' for context. For each experiment: modify strategy.py, commit, backtest, record to results.tsv, then decide whether to continue or exit. Follow the EXIT RULE in program-stateless.md exactly — you CANNOT exit without having attempted at least 2 architectural changes among your discards." \
    || {
      echo "Claude exited with error (code $?), continuing after cooldown..."
      sleep 5
    }
  echo "[DEBUG] Round $ROUND_COUNT completed (post-claude), looping..."

  # Ensure results.tsv ends with a newline (agent sometimes uses Write/Edit tool which strips it)
  if [ -f "$RESULTS" ] && [ -n "$(tail -c1 "$RESULTS")" ]; then
    echo >> "$RESULTS"
  fi

  echo ""
done
