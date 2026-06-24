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
commit	score	mean_score	std_score	bull_2021	crash_bear	sideways	rally_2024	mixed_2025	status	description
```

Use status `council_discard` or `council_keep` to distinguish from normal experiments.

## Scoring formula

Multiplicative per-regime score, then combined. This mirrors `compute_score()` in `prepare.py` plus the stability and flip-streak penalties applied in `regime_test.py` — it is the exact metric, not an approximation.

```
base = log(1+max(sharpe,0)) × sqrt(min(trades/50,1)) × dd_gate × exp(-streak/30) × log(1+max(APY,0)/10+1)
  dd_gate = 1/(1+DD%) × exp(-max(0,DD%-5)/2)   # soft@5%, scale=2; 0-5% mild, 5%+ steep
  APY = (1+total_return)^(8760/hours)-1   # annualized return; direct reward, NOT calmar

Per-regime score = base × stability_factor × flip_streak_gate
  stability_factor = clamp((stability-0.50)/(0.80-0.50), 0, 1)   # AR(1) correlated-noise test; applied only when base>0
  flip_streak_gate = 1/(1 + flip_streak_drag_per_bar/0.5)         # applied only when score>0

Hard cutoffs: <10 trades, >10% DD, >15% capital loss → -999
Soft DD penalty: smooth exponential above 5% DD (3%→0.97x, 5%→0.95x, 6%→0.74x, 10%→0.33x)

Composite = mean(regime_scores) - 0.3 * std(regime_scores)
```

**vol_gate removed (2026-06-19):** the former `1/(1+vol)` factor was a double penalty — `return_volatility` is the same std already in Sharpe's denominator. Near-constant 0.970-0.985 across all historical keeps (<1.4% regime spread), never changed ranking. Sharpe is now the sole vol-adjustment.

**std penalty lowered (2026-06-19):** `k` lowered 0.5 → 0.3. At k=0.5, ~72% of composite gains came from std reduction; agent over-optimized for consistency at the expense of mean return. k=0.3 keeps consistency reward (prevents abandoning weakest regime) while giving mean-improvement room.

**return_reward REPLACED by return_bonus (2026-06-24):** the prior `log(1+min(calmar,10)/10+1)` (calmar = APY/MaxDD) double-rewarded DD reduction — a DD drop raised BOTH dd_gate AND calmar (APY/DD), so the agent optimised "harvest to cut DD → calmar up" rather than "improve signal → return up". Measured: -1% DD was 3.8-4.6x more score-efficient than +0.1 Sharpe on bull/sideways; the 99a369a1 keep's +0.017 composite gain was 68-93% DD-driven (APY dropped). The new `return_bonus = log(1 + max(APY,0)/10 + 1)` rewards absolute annualized return directly. Under realistic margins (+0.2 Sharpe vs -0.5% DD vs +2% APY), Sharpe gain dominates DD reduction 11-36x on 4/5 regimes — pursue signal quality and return, not DD shaving. Leverage farming (raise LEVERAGE_K → APY and DD scale 1:1) is blocked by dd_gate knee@5 (below): any leverage increase pushes DD past 5% → exp penalty bites harder than linear APY bonus (verified: 1.1x leverage drops composite).

**dd_gate soft_start lowered 8→5 (2026-06-24):** the prior knee at 8% left a leverage-farming sweet spot — rally DD=5.16% could scale to 7.74% (LEVERAGE_K 6) before the exp penalty bit. Lowering the knee to 5% (where real cluster-regime DDs sit) makes any leverage increase bite immediately. This is the leverage-farming blocker that replaces the old calmar invariance (calmar is gone, so dd_gate alone must stop farming — knee@5 does it). Curve: DD=3%→0.97, DD=5%→0.95, DD=6%→0.74, DD=8%→0.55, DD=10%→0.33. Hard cutoff at 10% unchanged.

There is no return gate, turnover gate, or simplicity bonus — those were removed in the score-v5 overhaul. Higher trade frequency is NOT penalized beyond the fee-adjusted Sharpe already embedded in `base`.

The composite score is the key metric. Parse it from `grep "^composite_score:" run.log`.
