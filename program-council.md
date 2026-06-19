# Council Mode — adversarial convergence verification

You are running a **Council Mode session**. The normal experiment loop has detected convergence (N consecutive experiments with no improvement). Your job is to generate diverse adversarial proposals to either break through the plateau or confirm the strategy is near-optimal.

## Your task

1. **Read context**: Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline`.
2. **Analyze**: Identify the current best score, recent experiment history, and what has been tried.
3. **Generate 3-5 proposals**, each from a **distinct philosophy**:
   - **Simplification** — remove a component; test if performance holds without it
   - **Contrarian** — opposite of current approach (e.g., momentum → mean-reversion)
   - **Regime-shift** — what if market conditions shifted? (different vol regime, correlation breakdown)
   - **Scale-change** — different timeframe, asset weighting, or position sizing approach
   - **Radical** — completely different approach to the problem
4. **Anonymize & rank**: Label as Proposal A/B/C/D/E. Evaluate each on: pros, cons, overfitting risk, regime robustness, complexity cost. Output `FINAL RANKING: 1. Proposal X, 2. Proposal Y...`
5. **Execute in ranked order**: Apply #1, commit, backtest. If it improves score → keep and stop. If not → revert, try #2, then #3, etc.
6. **Output final verdict** (CRITICAL — the outer shell parses this):

If **any proposal improved** the score:
```
echo "COUNCIL_VERDICT: ACCEPT"
```

If **all proposals failed** to improve:
```
echo "COUNCIL_VERDICT: PASS"
```

You MUST output exactly one of these two lines as the very last thing before exiting.

## What you CAN do

- Modify `strategy.py` — this is the only file you edit.
- Run `uv run regime_test.py > run.log 2>&1` for each proposal.
- Commit and revert as needed. To revert a failed proposal, use `git revert --no-edit HEAD`. NEVER use `git reset --hard` — it destroys commits before the experiment.

## What you CANNOT do

- Modify `prepare.py`, `backtest.py`, `regime_test.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at holdout data (2025-01 onwards).
- Skip proposals — you must test at least 3 before declaring PASS.

## Council log

Append your session results to `council_log.md`:

```markdown
## Council Session #N — (score: <best_score>)

**Date:** <today>
**Trigger:** <N> consecutive no-improvement experiments
**Baseline:** score=<X>, sharpe=<Y>, dd=<Z>

### Proposals

| Label | Philosophy | Description |
|-------|-----------|-------------|
| A | ... | ... |
| B | ... | ... |
| C | ... | ... |

### Ranking

FINAL RANKING: 1. A, 2. C, 3. B — rationale: ...

### Results

| Proposal | Score | vs Baseline | Outcome |
|----------|-------|-------------|---------|
| A | ... | ... | discard/keep |
| C | ... | ... | discard/keep |
| B | ... | ... | discard/keep |

### Verdict

COUNCIL_PASS / COUNCIL_ACCEPT Proposal X (philosophy)
```

## Results TSV

Also record each proposal test in `results.tsv` using the 10-column schema (same as the normal experiment loop):
```
commit	score	mean_score	std_score	bull_2021	crash_bear	sideways	rally_2024	status	description
```

Use status `council_discard` or `council_keep` to distinguish from normal experiments.

## Scoring formula

Multiplicative per-regime score, then combined. This mirrors `compute_score()` in `prepare.py` plus the stability and flip-streak penalties applied in `regime_test.py` — it is the exact metric, not an approximation.

```
base = log(1+max(sharpe,0)) × sqrt(min(trades/50,1)) × dd_gate × exp(-streak/30)
  dd_gate = 1/(1+DD%) × exp(-max(0,DD%-5)/10)

Per-regime score = base × stability_factor × flip_streak_gate
  stability_factor = clamp((stability-0.50)/(0.80-0.50), 0, 1)   # AR(1) correlated-noise test; applied only when base>0
  flip_streak_gate = 1/(1 + flip_streak_drag_per_bar/0.5)         # applied only when score>0

Hard cutoffs: <10 trades, >10% DD, >15% capital loss → -999
Soft DD penalty: smooth exponential above 5% DD (5%→0.95x, 8%→0.68x, 10%→0.55x)

Composite = mean(regime_scores) - 0.3 * std(regime_scores)
```

**vol_gate removed (2026-06-19):** the former `1/(1+vol)` factor was a double penalty — `return_volatility` is the same std already in Sharpe's denominator. Near-constant 0.970-0.985 across all historical keeps (<1.4% regime spread), never changed ranking. Sharpe is now the sole vol-adjustment.

**std penalty lowered (2026-06-19):** `k` lowered 0.5 → 0.3. At k=0.5, ~72% of composite gains came from std reduction; agent over-optimized for consistency at the expense of mean return. k=0.3 keeps consistency reward (prevents abandoning weakest regime) while giving mean-improvement room.

There is no return gate, turnover gate, or simplicity bonus — those were removed in the score-v5 overhaul. Higher trade frequency is NOT penalized beyond the fee-adjusted Sharpe already embedded in `base`.

The composite score is the key metric. Parse it from `grep "^composite_score:" run.log`.
