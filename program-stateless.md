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

- Modify `prepare.py`, `backtest.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at test set data.

## Your single experiment

1. **Read context**: Read `strategy.py`, `results.tsv`, and run `git log main..HEAD --oneline` to see what has been tried on this branch.
2. **Analyze**: What worked? What failed? What hasn't been tried yet?
3. **Propose one change**: Pick one specific, testable idea. Prefer ideas that are different from recent experiments.
4. **Implement**: Edit `strategy.py` with your change.
5. **Commit**: `git commit -am "exp: <short description of what you changed>"`.
6. **Backtest**: `uv run backtest.py > run.log 2>&1`.
7. **Parse results**: `grep "^score:\|^sharpe:\|^max_drawdown_pct:\|^num_trades:" run.log`.
8. **Record**: If score improved vs the best in `results.tsv`, append a `keep` line. If score is worse or equal, run `git reset --hard HEAD~1` and append a `discard` line.
9. **Exit**: You are done. The outer loop will invoke you again for the next experiment.

## Results TSV format

```
commit	score	sharpe	max_dd	status	description
```

Append one line per experiment. Use the short commit hash, or `-` for discarded experiments.

## Scoring formula (from prepare.py)

```
score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty
trade_count_factor = min(num_trades / 50, 1.0)
drawdown_penalty = max(0, max_drawdown_pct - 15) * 0.05
turnover_penalty = max(0, annual_turnover/capital - 500) * 0.001
Hard cutoffs: <10 trades → -999, >50% drawdown → -999, lost >50% → -999
```

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
- Do NOT ask for confirmation. You are fully autonomous for this one experiment.
