# autotrader — multi-step experiment session

Autonomous trading strategy research on Hyperliquid perpetual futures.
You will run **up to 5 experiments** in this session, building on your findings iteratively. The outer shell script invokes you once per "round" — each round is a coherent research arc.

## Context

This project adapts Karpathy's autoresearch pattern for trading strategy discovery.
The owner has existing production strategies designed for tick-level market making (20-second intervals). Those strategies underperform when ported to hourly directional trading on this backtest harness.

Your job: **improve the current strategy in `strategy.py`** through iterative experimentation within a single session.

## What you CAN do

- Modify `strategy.py` — this is the only file you edit. Everything is fair game.

## What you CANNOT do

- Modify `prepare.py`, `backtest.py`, `regime_test.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at holdout data (2025-01 onwards).

### Phase priority rule
When min_stability < 0.80: at least 3 of 5 experiments MUST target stability (use stability keep path). Remaining 2 may target composite.

## Session protocol

You run up to 5 experiments per session. Each experiment may be a **single-variable change OR a multi-variable structural change** — whichever is appropriate for the hypothesis. Multi-variable changes are especially encouraged for stability improvements, where architectural modifications (voter weighting, signal fusion, ensemble method changes) inherently require coordinated edits. You can **use insights from earlier experiments in this session to choose your next direction**, and after step 2 you may attempt **combination experiments** that merge two independently-validated improvements.

### Phase 1: Read context (once per session)

1. Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline -n 30`.
2. **Analyze**: What worked? What failed? What hasn't been tried? **Saturation check**: grep `results.tsv` descriptions for directions you're considering. If a direction has 10+ prior experiments with mostly `discard`, it's saturated — switch to something structurally different.

### ESCALATION RULE (multi-variable architectural change)

If the last 5+ stability-targeted experiments in `results.tsv` all achieved < +0.005 stability gain, single-parameter threshold tuning is **exhausted**. Your next stability experiment MUST be a **multi-variable architectural change** — e.g., replace binary voting with weighted ensemble, restructure exit logic, change signal fusion method, add hysteresis layers, or redesign position sizing. Do NOT repeat incremental threshold/parameter tweaks that have been proven to plateau.

### Phase 2: Experiment loop (repeat up to 5 times)

For each experiment:

1. **Propose one change**: Pick one specific, testable idea. After your first experiment in this session, you may base your next idea on the regime-level insights you just observed (e.g., "Exp A showed sideways +0.24 but crash -1.44 — try a different condition that protects crash").
2. **Implement**: Edit `strategy.py` with your change.
3. **Commit**:
   ```
   git commit -m "exp: <short description of what you changed>" \
              -m "Hypothesis: <1-2 sentences on the mechanism>" \
              -m "Expected: <which regime(s) should benefit>"
   ```
4. **Backtest**: `uv run regime_test.py > run.log 2>&1`
5. **Parse results**: `grep "^composite_score:\|^mean_score:\|^std_score:\|^regime_\|^min_stability:" run.log`
6. **Record** (mandatory — do NOT skip): Append one row to `results.tsv` for EVERY experiment. This is not optional.

   **Keep/discard rules (composite path):**
   - `composite_score` improved by **at least +0.03** vs the best `keep` in `results.tsv`.
   - No individual `regime_score` regressed by more than **`max(0.2, 5 × composite_gain)`** vs baseline.
   - **No more than 2 out of 4 regimes may regress** (strictly negative Δ).
   - **While `min_stability < 0.80`:** composite keep also requires `min_stability` did NOT decrease vs baseline (Δ ≥ 0). Composite improvements that sacrifice stability are rejected.

   **Stability keep (alternative path):** When `min_stability < 0.80`, an experiment qualifies as keep if:
   - `min_stability` improved by **at least +0.01** vs baseline.
   - No regime's `max_dd_pct` increased by more than **1.0%** vs baseline.
   - `composite_score` did NOT drop by more than **2.0** vs baseline.
   - This path exists because structural changes that improve stability often reduce returns (fewer entries, smaller positions, smoother signals). That's acceptable — crossing 0.80 removes the 50% penalty, which more than compensates for moderate return loss. But DD must not worsen.

   If keep: append a `keep` line with all per-regime scores. The new baseline for subsequent experiments in this session is now this keep.
   If discard: run `git revert --no-edit HEAD`, append a `discard` line. NEVER use `git reset --hard`.

7. **Decide next step**:
   - If you have a clear follow-up insight from the regime breakdown → continue to next experiment.
   - If you've found a keep and want to try combining it with another idea → continue.
   - If you've exhausted your ideas or hit 5 experiments → exit.

### Phase 3: Combination experiments (optional, experiments 3-5)

After running at least 2 independent experiments and observing their regime-level effects, you MAY attempt a combination:
- **Prerequisite**: at least one of the component ideas showed a promising signal (e.g., improved a target regime even if overall was discard due to regression elsewhere).
- **Combination = applying two ideas together** in a single `strategy.py` edit. Still counts as one experiment, still gets one results.tsv row.
- **Attribution**: in the commit message, reference which prior experiments you're combining (e.g., "Combines the EMA spread filter (sideways +0.24) with vol_ratio gating (crash protection)").
- **Same keep/discard rules apply** — no special treatment for combinations.

### Session end

After your last experiment (or when you hit 5), exit. The outer loop will invoke you again for the next round.

### Regime gate rationale

The regime gates prevent regime-fit experiments that trade one regime for another. The **magnitude cap** auto-scales: at minimum keep threshold (composite_gain=+0.01) the cap is 0.2; at larger gains (composite_gain=+0.1 → cap 0.5) modest rebalancing is tolerated. The **majority rule** catches experiments where no single regression is large but most regimes drift down (std-gaming, not real alpha).

## Results TSV format

New schema (10 columns, tab-separated):
```
commit	score	mean_score	std_score	bull_2021	crash_bear	sideways	rally_2024	status	description
```

Legacy rows (6 columns) may remain in the file for historical reference but are ignored when computing the per-regime baseline. Always append new rows using the 10-column schema.

- `score` = composite_score (mean - 0.5*std)
- `mean_score` = average across 4 regimes
- `std_score` = std across regimes (lower = more consistent)
- `bull_2021 / crash_bear / sideways / rally_2024` = per-regime scores extracted from lines matching `^regime_<name>_score:` in run.log (e.g., `regime_bull_2021_score: 27.123456` → store `27.12`)
- Append one line per experiment. Use the short commit hash, or `-` for discarded.

## Scoring formula

Each regime is scored via multiplicative `compute_score()`, then combined:

```
Base score = log(1+sharpe)         # signal quality
           × sqrt(trade_factor)    # sample sufficiency
           × 1/(1 + DD%)           # base drawdown gate
           × exp(-max(0, DD%-5)/10) # soft DD penalty (mild slope above 5%)
           × 1/(1 + vol)           # volatility gate
           × exp(-streak/30)       # consecutive loss gate

Per-regime score = base_score × log(1 + annual_return% / 100)   # return gate

Hard cutoffs: <10 trades → -999, >20% drawdown → -999, lost >25% → -999

Composite score = mean(regime_scores) - 0.5 * std(regime_scores) + simplicity_bonus
Simplicity bonus = max(0, (500 - effective_LOC)) * 0.001   # reward shorter strategy.py
```

The simplicity bonus rewards removing dead code and unnecessary complexity. Effective LOC counts non-empty, non-comment lines in strategy.py. Each line removed below 500 adds +0.001 to composite.

Multiplicative structure: any dimension being terrible collapses the entire score.
The DD penalty is a smooth exponential — no cliff at any specific DD level. DD 5%→no penalty, 8%→0.74x, 10%→0.61x, 15%→0.37x.
The return gate prevents gaming via position-size reduction (smaller positions improve DD/vol gates but reduce returns).
The composite rewards strategies that perform **consistently across all market conditions**.

Search regimes (4 non-overlapping periods):
- bull_2021: 2021-01 ~ 2021-10 (bull market)
- crash_bear: 2021-11 ~ 2022-12 (Luna/FTX crash + deep bear)
- sideways: 2023-01 ~ 2023-12 (sideways recovery)
- rally_2024: 2024-01 ~ 2024-12 (ETF + election rally)

## Primary Objective: Signal Stability (min_stability ≥ 0.85)

**Stability is the #1 priority.** The scoring now applies a **hard 50% penalty** when stability < 0.80. This means:
- stability 0.70 → factor = (0.70/0.85) × 0.5 = 0.41 (score loses 59%)
- stability 0.80 → factor = 0.80/0.85 = 0.94 (score loses only 6%)
- stability 0.85+ → factor = 1.0 (no penalty)

Crossing the 0.80 threshold is worth ~+40% on every regime score simultaneously. This dwarfs any parameter tweak.

**Do NOT conclude that "stability requires fundamentally different architecture and is too risky."** That reasoning is a trap — it leads to endless base-performance tweaks that never close the gap. Structural changes to improve stability ARE the highest-ROI experiments available.

**Multi-variable structural changes are explicitly allowed** for stability work. You are NOT limited to single-parameter tweaks. Diagnose the noise sensitivity source first, then propose whatever scope of change is needed — including architectural modifications that touch multiple components simultaneously.

### Diagnostic-first approach (mandatory before proposing stability fixes)

Before proposing any solution, **diagnose** where the noise sensitivity actually comes from:
1. Read `strategy.py` and identify all voters/signals that use hard thresholds on price-derived values
2. For each voter, estimate how far typical signal values sit from their decision boundary — voters that hover near threshold are the primary noise amplifiers
3. Propose changes that target the MOST sensitive voter/threshold first, not broad architectural rewrites

### How to evaluate stability experiments
- Check `regime_X_stability` in the output — ALL four should improve toward 0.85+
- A stability gain of +0.05 (e.g., 0.70→0.75) is worth pursuing even if base_score drops slightly (the net effect on composite depends on the trade-off)
- Acceptable trade: lose ≤2.0 base_score if stability improves by ≥0.10 (net composite gain from reduced penalty)

## Stability-first directions (priority when min_stability < 0.80)

**Do NOT use open price as a "stable" signal source.** The noise test only perturbs close (then adjusts high/low). Open appears noise-immune but this is an artifact of the test methodology, not a real property. In live trading, open is equally noisy. Any stability gain from using open is illusory and will not generalize.
**HL2 stability gains are overstated.** HL2=(high+low)/2 receives roughly half the perturbation of close (because high/low only change when perturbed close exceeds original range). In trending regimes with wide bars, HL2 is nearly unperturbed — this flatters stability scores. Acceptable use: multi-point aggregations (e.g., linreg over 16 bars) where averaging further reduces noise. Unacceptable use: single-point comparisons (e.g., Donchian max/min) or magnitude calculations (breaks sizing calibration). Always discount reported HL2 stability gains by ~50%.

### Stability methodology: diagnose → layered defense

**⚠️ HARD RULE: Any stability experiment attempted WITHOUT first completing Step 1 diagnosis is INVALID. You must have concrete flip-rate numbers before proposing a fix. "I think X is noisy" is not diagnosis — you need measured data.**

**Step 1: Diagnose the weakest voter (MANDATORY — run this code before ANY stability experiment)**

Run this diagnostic ONCE at the start of each session (before your first stability experiment). Copy this into a temporary script, execute it, then delete the script:

```python
# diagnostic: per-voter flip rate under ±5bps noise
# Add to strategy.py temporarily, run via: uv run python -c "from strategy import diagnose_flips; diagnose_flips()"
def diagnose_flips():
    """Compute per-voter flip rate: how often does each voter change output under ±5bps close perturbation?"""
    import numpy as np
    from prepare import load_data
    data = load_data()
    # Use the longest available symbol
    for sym in ['BTC', 'ETH']:
        if sym not in data: continue
        df = data[sym]
        closes_clean = df['close'].values.copy()
        n_bars = len(closes_clean)
        n_trials = 20
        noise_mag = 0.0005  # ±5bps
        
        # For each bar, run strategy on clean vs perturbed, record each voter's output
        # You must instrument the vote computation to extract individual voter booleans
        # Compare: voter_bull_clean[i] != voter_bull_perturbed[i] → flip
        # Report: flip_rate[voter] = flips / (n_bars * n_trials)
        print(f"TODO: instrument {sym} voters, compute flip rates")
        print(f"Target: identify voter with flip_rate > 0.30")
diagnose_flips()
```

Adapt this skeleton to actually instrument your voters (extract each voter's boolean output per bar under clean vs perturbed close). The output you need is a table like:
```
Voter          | Flip rate | Bars affected
ema_cross      | 0.12      | 847
macd_hist      | 0.08      | 592  
donchian       | 0.31      | 2184  ← PRIMARY TARGET
linreg_slope   | 0.05      | 350
...
```

**Only after you have this table** may you proceed to Step 2.

**Step 2: Choose intervention from four layers (address the outermost broken layer first)**

| Layer | What it does | Example approaches |
|-------|-------------|-------------------|
| 1. Input denoising | Remove noise before indicators see it | Pre-filter close with low-pass/robust smoother before feeding to voters |
| 2. Robust indicator | Make the indicator computation itself noise-resistant | Robust regression (median-based), Kalman velocity with uncertainty, longer aggregation windows |
| 3. Per-voter hysteresis | Prevent individual voter output from flipping on small moves | Asymmetric enter/exit thresholds, minimum hold time before voter can flip |
| 4. Aggregate margin | Make the collective decision robust even if individual voters flip | Require margin (not just majority), confidence-weighted voting (distance from boundary = weight), abstain zone |

Most stability improvements come from layers 3 and 4 (cheapest to implement, highest impact on boolean-voter architectures). Layer 1-2 changes are more invasive but have higher ceiling.

**Step 3: Two valid paths to higher stability**
- **Removal/simplification**: eliminate the noisiest voter entirely. If stability jumps +0.02+ even with composite loss, that confirms it was a noise source. This is how vol_breakout removal worked.
- **Structural change**: add hysteresis, confidence weighting, or input denoising to make existing voters more robust without removing them. This preserves signal diversity.

Both are valid. Diagnose first, then choose based on the flip rate magnitude.

Do NOT hardcode "proven ineffective" conclusions here — read results.tsv each round to discover what has been tried. Only methodology-level blind spots (like open price artifact and HL2 overestimation above) belong in this file.

## Strategy research directions

Start with these high-probability ideas:

### Tier 1 — Most likely to improve score
- **Add SOL with lower weight** — diversification should help Sharpe
- **Vol-regime adaptive sizing** — reduce positions in high vol, increase in low vol
- **Multi-timeframe momentum** — require 12h, 24h, 48h agreement before entry
- **ATR-based trailing stops** — volatility-adjusted trailing exits
- **Funding carry overlay** — add carry component on top of momentum

### Tier 2 — Worth exploring
- **EMA crossover instead of raw momentum** — smoother signals, fewer whipsaws
- **Cross-asset lead-lag** — BTC momentum predicts ETH/SOL 1-6h later
- **Dynamic threshold** — adjust momentum entry threshold by recent vol
- **Inverse vol position sizing** — proven in production risk framework
- **Ensemble voting** — combine 3+ signals, only enter when majority agree

### Structural issues (code review findings)
The sideways regime has the lowest score (20.32 vs bull 32.45). These code-structural issues may explain why:
- **Cooldown vanishes in low-trend**: `effective_cooldown = COOLDOWN_BARS * min(|ret_long|/0.06, 1.0)` drops to ~0.5 bars when `|ret_long|` is small. A minimum floor (e.g. `max(1, ...)`) would prevent degenerate re-entry cycles without affecting trending regimes.
- **Mean-reversion RSI 49/51 covers ~95% of RSI values**: RSI oscillates around 50 by construction, so these thresholds make mean-rev the dominant entry path in sideways, overriding voting-based momentum logic.
- **Momentum and mean-reversion share one cooldown**: both entry paths gate on the same `in_cooldown` flag. Decoupling them (e.g. fixed cooldown for mean-rev entries) could prevent sideways churn.

### Tier 3 — Radical / novel
- **Pure mean reversion on funding rate** — trade the mean reversion of funding itself
- **Correlation regime switching** — different strategies for high/low BTC-ETH correlation
- **Pairs trading** — long ETH/short BTC (or vice versa) on relative value
- **Time-of-day patterns** — are there hourly seasonality patterns?
- **Volatility breakout** — enter when realized vol breaks above/below its own SMA
- **Machine learning lite** — rolling linear regression of features → direction

## Data available

- BTC, ETH, SOL hourly OHLCV + funding rates
- History buffer: last 500 bars via `bar_data[symbol].history` DataFrame
- Columns: timestamp, open, high, low, close, volume, funding_rate

## Overfitting hygiene

These rules exist because this branch has accumulated 190+ experiments. At that scale, selection bias dominates — any single +0.01 improvement is statistically fragile, and the search regimes themselves are effectively in-sample. Violating these rules causes meta-overfit that the regime-regression gate cannot catch.

- **Do NOT read commit bodies of prior experiments.** Use only `git log main..HEAD --oneline` (subjects only) and `results.tsv`. Commit bodies hold past hypotheses — reading them narrows your proposal space to "slight variants of what was tried," amplifying selection bias.
- **Do NOT base your idea on holdout findings.** The holdout (2025-01+) is never read by you. But also: if you notice phrasing in this prompt or in code comments that references specific holdout events (e.g., "the single-hour DD on 2025-03-02"), treat those as off-limits — do NOT design a rule targeting them. Any insight derived from holdout data is information leakage, regardless of who performed the analysis.
- **Prefer mechanism-backed changes over parameter sweeps.** At this experiment count, moving a parameter by 10% and finding +0.01 is likely noise. Adding a new signal with a clear mechanism, or removing a component that should be redundant, is a more honest trial.
- **Respect saturation signals.** If step 2 finds a direction has 10+ prior tries with mostly discards, do not submit another tuning variant.

## Guidelines

- **One variable per experiment** (within a session you run multiple experiments, but each tests one idea in isolation). The exception is Phase 3 combination experiments, which explicitly merge two independently-validated ideas.
- If you have no ideas, re-read `strategy.py` carefully and look for parameters to tune or signals to add/remove.
- All else equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code is not worth it.
- **Simplification experiments are as valuable as additions.** Try removing a voter, disabling a sizing multiplier, or deleting dead code. If the score holds or improves, keep the simpler version. Complexity has a hidden cost: it hurts out-of-sample generalization.
- **Do NOT inline constants or compress code for LOC bonus.** Named constants improve readability. The simplicity bonus rewards removing dead logic, not cosmetic code compression. Inlining a named constant into its usage site is NOT a valid simplification.
- **Use your session context wisely.** Your advantage over single-experiment mode is that you can observe patterns across experiments within this session. If experiment 1 shows sideways +0.24 but crash -1.44, experiment 2 should try to preserve the sideways gain while protecting crash — not start from scratch on an unrelated direction.
- Do NOT ask for confirmation. You are fully autonomous for this session.
