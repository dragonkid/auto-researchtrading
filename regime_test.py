"""
Regime robustness test: run the strategy across different market regimes.
Computes a composite score = mean(scores) - k*std(scores) to reward
strategies that work across ALL market conditions.

Usage: uv run regime_test.py
"""

import math
import time
import signal as sig

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET
from strategy import Strategy

# Non-overlapping regimes for parameter search
# These cover 4 distinct market conditions across 4 years
SEARCH_REGIMES = [
    ("bull_2021", "2021-01-01", "2021-10-31", "Bull market / main uptrend"),
    ("crash_bear", "2021-11-01", "2022-12-31", "Luna + FTX crash / deep bear"),
    ("sideways", "2023-01-01", "2023-12-31", "Sideways recovery"),
    ("rally_2024", "2024-01-01", "2024-12-31", "ETF + election rally"),
]

# Holdout regime — NEVER used during autoresearch search.
# Only run manually for final validation after a research round.
HOLDOUT_REGIMES = [
    ("recent", "2025-01-01", "2026-03-31", "Recent market (holdout)"),
]

# Consistency penalty weight: higher k = stricter consistency requirement
CONSISTENCY_K = 0.5


def timeout_handler(signum, frame):
    print("TIMEOUT: backtest exceeded time budget")
    raise TimeoutError


def run_regime(name: str, start: str, end: str, desc: str) -> dict:
    sig.signal(sig.SIGALRM, timeout_handler)
    sig.alarm(TIME_BUDGET + 60)

    strategy = Strategy()
    data = load_data(start=start, end=end)

    total_bars = sum(len(df) for df in data.values())
    if total_bars == 0:
        sig.alarm(0)
        return {"name": name, "desc": desc, "bars": 0, "error": "no data"}

    result = run_backtest(strategy, data)
    score = compute_score(result)

    sig.alarm(0)
    return {
        "name": name,
        "desc": desc,
        "bars": total_bars,
        "score": score,
        "sharpe": result.sharpe,
        "return_pct": result.total_return_pct,
        "max_dd_pct": result.max_drawdown_pct,
        "trades": result.num_trades,
        "win_rate": result.win_rate_pct,
        "profit_factor": result.profit_factor,
        "seconds": result.backtest_seconds,
    }


def compute_composite_score(results: list[dict]) -> float:
    """Composite = mean(scores) - k * std(scores). Returns -999 if any regime failed."""
    scores = []
    for r in results:
        if "error" in r or r.get("score", -999) <= -999:
            return -999.0
        scores.append(r["score"])

    if not scores:
        return -999.0

    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = math.sqrt(variance)

    return mean_score - CONSISTENCY_K * std_score


if __name__ == "__main__":
    import sys

    # --holdout flag runs holdout regimes instead of search regimes
    use_holdout = "--holdout" in sys.argv
    regimes = HOLDOUT_REGIMES if use_holdout else SEARCH_REGIMES

    if use_holdout:
        print("=== HOLDOUT VALIDATION (not for autoresearch) ===\n")

    results = []
    for name, start, end, desc in regimes:
        print(f"Running: {name} ({start} ~ {end})...")
        t0 = time.time()
        try:
            r = run_regime(name, start, end, desc)
        except (TimeoutError, Exception) as e:
            r = {"name": name, "desc": desc, "bars": 0, "error": str(e)}
        elapsed = time.time() - t0
        r["wall_time"] = elapsed
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Sharpe: {r['sharpe']:>8.2f}  Return: {r['return_pct']:>+8.1f}%  MaxDD: {r['max_dd_pct']:>6.2f}%  Score: {r['score']:>8.2f}")
        print()

    # Summary table
    print("=" * 120)
    print(f"{'Regime':<15} {'Period':<25} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Score':>8}")
    print("-" * 120)
    for (name, start, end, desc), r in zip(regimes, results):
        if "error" in r:
            print(f"{name:<15} {start}~{end}  {'ERROR':>8}  {r.get('error', '')}")
        else:
            print(
                f"{name:<15} {start}~{end}"
                f"  {r['sharpe']:>8.2f}"
                f"  {r['return_pct']:>+8.1f}%"
                f"  {r['max_dd_pct']:>7.2f}%"
                f"  {r['trades']:>6}"
                f"  {r['win_rate']:>6.1f}%"
                f"  {r['profit_factor']:>5.1f}"
                f"  {r['score']:>8.2f}"
            )
    print("=" * 120)

    # Composite score (only for search regimes)
    if not use_holdout:
        composite = compute_composite_score(results)
        scores = [r["score"] for r in results if "error" not in r]
        mean_s = sum(scores) / len(scores) if scores else 0
        var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores) if scores else 0
        std_s = math.sqrt(var_s)

        # Parseable output for autoresearch agent
        print("---")
        print(f"composite_score:    {composite:.6f}")
        print(f"mean_score:         {mean_s:.6f}")
        print(f"std_score:          {std_s:.6f}")
        print(f"num_regimes:        {len(scores)}")
        for r in results:
            if "error" not in r:
                print(f"regime_{r['name']}_score: {r['score']:.6f}")
                print(f"regime_{r['name']}_sharpe: {r['sharpe']:.6f}")
                print(f"regime_{r['name']}_return_pct: {r['return_pct']:.6f}")
                print(f"regime_{r['name']}_max_dd: {r['max_dd_pct']:.6f}")
