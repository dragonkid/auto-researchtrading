"""
Regime robustness test: run the strategy across different market regimes.
Usage: uv run regime_test.py
"""

import time
import signal as sig

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET
from strategy import Strategy

REGIMES = [
    ("val (optimized)", "2024-07-01", "2025-03-31", "Bull — strategy optimized here"),
    ("test", "2025-04-01", "2025-12-31", "Bull → sideways"),
    ("2022_crash", "2022-04-01", "2022-12-31", "Luna + FTX crash"),
    ("2023_sideways", "2023-01-01", "2023-09-30", "Sideways recovery"),
    ("recent_bear", "2025-10-01", "2026-03-31", "Recent bear market"),
    ("full_bear", "2021-11-01", "2023-01-31", "Full bear cycle ATH→bottom"),
]


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


if __name__ == "__main__":
    results = []
    for name, start, end, desc in REGIMES:
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
            print(f"  Sharpe: {r['sharpe']:>8.2f}  Return: {r['return_pct']:>+8.1f}%  MaxDD: {r['max_dd_pct']:>6.2f}%  Trades: {r['trades']}")
        print()

    print("=" * 110)
    print(f"{'Regime':<20} {'Period':<25} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Score':>8}")
    print("-" * 110)
    for (name, start, end, desc), r in zip(REGIMES, results):
        if "error" in r:
            print(f"{name:<20} {start}~{end}  {'ERROR':>8}  {r.get('error', '')}")
        else:
            print(
                f"{name:<20} {start}~{end}"
                f"  {r['sharpe']:>8.2f}"
                f"  {r['return_pct']:>+8.1f}%"
                f"  {r['max_dd_pct']:>7.2f}%"
                f"  {r['trades']:>6}"
                f"  {r['win_rate']:>6.1f}%"
                f"  {r['profit_factor']:>5.1f}"
                f"  {r['score']:>8.2f}"
            )
    print("=" * 110)
