# Auto-Research Trading

Karpathy-style autoresearch for Hyperliquid perpetual futures.
AI agent autonomously modifies `strategy.py`, backtests, keeps improvements.

## Commands

```bash
uv run prepare.py          # Download data (cached to ~/.cache/autotrader/data/)
uv run backtest.py         # Run current strategy, prints score
uv run run_benchmarks.py   # Compare 5 reference strategies
```

## Project Structure

- `strategy.py`     — THE ONLY MUTABLE FILE. Agent edits this.
- `prepare.py`      — Data + backtest engine + scoring. FROZEN.
- `backtest.py`     — Entry point. FROZEN.
- `program.md`      — Agent instructions for autonomous loop. Human-written.
- `benchmarks/`     — 5 reference strategies. FROZEN.
- `STRATEGIES.md`   — Full 103-experiment evolution log.

## Key Constraints

- Only edit `strategy.py` (implements `Strategy.on_bar()`)
- Allowed deps: numpy, pandas, scipy, requests, pyarrow, stdlib only
- Backtest time budget: 120 seconds
- Data: BTC/ETH/SOL hourly OHLCV + funding rate, 2024-07 ~ 2025-03
- Initial capital: $100,000

## Scoring

```
score = sharpe * sqrt(min(trades/50, 1.0)) - drawdown_penalty - turnover_penalty
Hard cutoffs (-> -999): <10 trades, >50% drawdown, >50% capital loss
```

## Current Best

Score 20.634 (Sharpe 20.634, 0.3% max DD, 7605 trades) on branch `autotrader/mar10c`.
Baseline to beat: 2.724 (simple_momentum).

## Branches

- `main` — Base scaffold and data pipeline
- `autotrader/*` — Experiment branches (`mar10c` is best)

## Analysis Findings

- Sharpe 21.4 是真实回测结果，无膨胀机制（Sharpe 是 scale-invariant 的）
- 高 Sharpe 来源：单笔 Sharpe 0.61 × √10000 年交易次数 = √N 放大
- $3k 交易量下 6bps 摩擦合理（Hyperliquid 实际费率更低）
- Agent 砍仓位（0.16→0.08）是 gaming Score 的 turnover penalty，非 Sharpe
- 测试集 Sharpe 18.5（衰减 14%），核心风险是 regime 依赖 + 参数过拟合
- 6 个信号全部是趋势跟踪指标，在横盘/熊市中会失效

## TODO

- [ ] Regime robustness 验证：爬最近 6 个月数据（2025-10 ~ 2026-03，熊市+横盘）重跑回测
  - 需要修改 `prepare.py` 的 `TEST_END` 或新增一个 `recent` split
  - 删除 `~/.cache/autotrader/data/*.parquet` 强制重新下载（现有 parquet 不含新时段数据）
  - 预期：如果策略是 regime-dependent，Sharpe 应从 18-21 降到 < 3
- [ ] 用 train set（2023-06 ~ 2024-06，跨越熊转牛）回测，测试 regime 切换下的表现
- [ ] 评分函数改进实验：添加换手率下限惩罚，观察 agent 在新函数下的行为变化

## Gotchas

- Data downloads on first run (~1 min), then cached at `~/.cache/autotrader/data/`
- `backtest.py` has SIGALRM timeout (budget + 30s grace) — strategy must be fast
- `program.md` is the autoresearch harness prompt, not a dev doc
- Equity curve CSVs and `charts/` are pre-generated artifacts, not live outputs
