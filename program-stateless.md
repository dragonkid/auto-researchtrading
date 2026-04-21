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

1. **Read context**: Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline` to see what has been tried on this branch.
2. **Analyze**: What worked? What failed? What hasn't been tried yet?
3. **Propose one change**: Pick one specific, testable idea. Prefer ideas that are different from recent experiments.
4. **Implement**: Edit `strategy.py` with your change.
5. **Commit**: `git commit -am "exp: <short description of what you changed>"`.
6. **Backtest**: `uv run regime_test.py > run.log 2>&1`. This runs backtests across 4 non-overlapping market regimes (bull, bear crash, sideways, rally) and outputs a composite score.
7. **Parse results**: `grep "^composite_score:\|^mean_score:\|^std_score:\|^regime_" run.log`. The key metric is `composite_score` (= mean - 0.5*std across regimes). Also check individual regime scores for insights.
8. **Record**: If `composite_score` improved vs the best in `results.tsv`, append a `keep` line. If worse or equal, run `git revert --no-edit HEAD` to fully undo the experiment commit (preserves all prior commits including harness files), then append a `discard` line. NEVER use `git reset --hard` — it destroys commits before the experiment.
9. **Exit**: You are done. The outer loop will invoke you again for the next experiment.

## Results TSV format

```
commit	score	mean_score	std_score	status	description
```

- `score` = composite_score (mean - 0.5*std)
- `mean_score` = average across 4 regimes
- `std_score` = std across regimes (lower = more consistent)
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
- **ATR-based trailing stops** — current fixed stops are suboptimal
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

## Guidelines

- One change per experiment. Keep it atomic so you know what caused the score change.
- If you have no ideas, re-read `strategy.py` carefully and look for parameters to tune or signals to add/remove.
- All else equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code is not worth it.
- **Simplification experiments are as valuable as additions.** Try removing a voter, disabling a sizing multiplier, or deleting dead code. If the score holds or improves, keep the simpler version. Complexity has a hidden cost: it hurts out-of-sample generalization.
- **Funding rate data is all zeros** — do not waste experiments on funding-related features.
- **ATR trailing stop never triggers** — decel/RSI exits always fire first. Do not tune ATR stop parameters.
- Do NOT ask for confirmation. You are fully autonomous for this one experiment.
