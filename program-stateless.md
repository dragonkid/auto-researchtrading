# autotrader — single experiment

Autonomous trading strategy research on Hyperliquid perpetual futures.
You will run **one experiment**, then exit. The outer shell script handles looping.

## Context

This project adapts Karpathy's autoresearch pattern for trading strategy discovery.
The owner has existing production strategies designed for tick-level market making (20-second intervals). Those strategies underperform when ported to hourly directional trading on this backtest harness.

Your job: **improve the current strategy in `strategy.py`** by trying one experimental idea per invocation.

## What you CAN do

- Modify `strategy.py` — this is the only file you edit. Everything is fair game.

## What you CANNOT do

- Modify `prepare.py`, `backtest.py`, `regime_test.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at holdout data (2025-01 onwards).

## Your single experiment

1. **Read context**: Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline -n 30` to see recent activity on this branch. `results.tsv` is the canonical experiment log — use it for historical context. Expand the git log range only if you need to verify a specific older experiment that `results.tsv` references.
2. **Analyze**: What worked? What failed? What hasn't been tried yet? **Saturation check**: grep `results.tsv` descriptions for the direction you're considering (e.g., "sideways", "peak_profit", "cooldown"). If that direction has 10+ prior experiments and the recent ones are mostly `discard`, the direction is saturated. Do NOT submit another tuning variant — switch to a structurally different idea (new signal, removed component, different regime target).
3. **Propose one change**: Pick one specific, testable idea. Prefer ideas that are different from recent experiments. Prefer changes backed by a concrete mechanism (a reason it should work) over pure parameter sweeps — at N ~200 experiments, pure sweeps are statistically fragile under multiple-testing.
4. **Implement**: Edit `strategy.py` with your change.
5. **Commit**: Subject line stays short; put the hypothesis in the commit BODY.
   ```
   git commit -m "exp: <short description of what you changed>" \
              -m "Hypothesis: <1-2 sentences on the mechanism — why should this improve composite_score?>" \
              -m "Expected: <which regime(s) should benefit, e.g. 'bull_2021 via reduced whipsaw; others neutral'>"
   ```
   The body is for **post-hoc human audit only**. It is NOT input for future agents (see overfitting hygiene section below).
6. **Backtest**: `uv run regime_test.py > run.log 2>&1`. This runs backtests across 4 non-overlapping market regimes (bull, bear crash, sideways, rally) and outputs a composite score.
7. **Parse results**: `grep "^composite_score:\|^mean_score:\|^std_score:\|^regime_" run.log`. The key metric is `composite_score` (= mean - 0.5*std across regimes). Also check individual regime scores for insights.
8. **Record** (mandatory — do NOT skip): Every experiment, regardless of outcome, MUST produce exactly one new row appended to `results.tsv` before you exit. This is not optional. On run1, only ~9% of ~700 attempted experiments were logged — the missing 91% silently inflated selection bias (the true N was hidden, so multiple-testing math was under-estimated, and kept experiments looked more significant than they were). Do not repeat that failure mode on run2.

   Keep the experiment ONLY IF BOTH conditions hold:
   - `composite_score` improved by **at least +0.01** vs the best `keep` in `results.tsv` (improvements below +0.01 are noise at score ~24 — treat as discard).
   - No individual `regime_score` regressed by more than **`max(0.2, 5 × composite_gain)`** vs the baseline, where `composite_gain = new_composite - baseline_composite`. Baseline = the most recent `keep` line in `results.tsv` that has per-regime columns (10-column schema, see below). If no such line exists yet (first run after schema adoption), skip the regime-regression check.
   - **No more than 2 out of 4 regimes may regress** (have `Δregime_score < 0` vs baseline). Count strictly negative deltas; if 3 or 4 regimes regress, reject regardless of the single-regime magnitude. This catches regime-trade experiments whose composite improvement comes from std reduction (one regime gains, most lose slightly) rather than broad-based alpha.

   Rationale for the regime gates: prevent regime-fit experiments that trade one regime for another. The **magnitude cap** (first rule) auto-scales with gain size: at minimum keep threshold (composite_gain=+0.01) the cap is 0.2, so any meaningful single-regime loss is rejected; at larger gains (composite_gain=+0.1 → cap 0.5) modest regime rebalancing is tolerated. The **majority rule** (second rule) is orthogonal — it catches experiments where no single regression is large but most regimes still drift down. Example of what the magnitude cap catches: composite +0.02 driven by +0.5 in one regime and -0.4 in another (the 0.4 exceeds the 0.2 cap). Example of what the majority rule catches: composite +0.02 driven by +0.17 in one regime and -0.02/-0.12/-0.03 in the other three (no single regression exceeds the 0.2 cap, but 3/4 regimes are down — this is std-gaming disguised as improvement, not real alpha).

   If both conditions hold, append a `keep` line with all per-regime scores. Otherwise run `git revert --no-edit HEAD` to fully undo the experiment commit (preserves all prior commits including harness files), then append a `discard` line. NEVER use `git reset --hard` — it destroys commits before the experiment.
9. **Exit**: You are done. The outer loop will invoke you again for the next experiment.

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

- One change per experiment. Keep it atomic so you know what caused the score change.
- If you have no ideas, re-read `strategy.py` carefully and look for parameters to tune or signals to add/remove.
- All else equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code is not worth it.
- **Simplification experiments are as valuable as additions.** Try removing a voter, disabling a sizing multiplier, or deleting dead code. If the score holds or improves, keep the simpler version. Complexity has a hidden cost: it hurts out-of-sample generalization.
- **Do NOT inline constants or compress code for LOC bonus.** Named constants improve readability. The simplicity bonus rewards removing dead logic, not cosmetic code compression. Inlining a named constant into its usage site is NOT a valid simplification.
- Do NOT ask for confirmation. You are fully autonomous for this one experiment.
