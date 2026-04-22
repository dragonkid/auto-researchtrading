"""
Holdout sub-period analysis: split the 2025-01 ~ 2026-03 holdout into smaller
windows to identify when the strategy performs poorly.

Reports:
  - Quarterly breakdown (5 quarters)
  - Rolling 30-day windows (for finer resolution)

Usage: uv run holdout_subperiod.py
"""

import math
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from datetime import date, timedelta

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET


QUARTERS = [
    ("2025-Q1", "2025-01-01", "2025-03-31"),
    ("2025-Q2", "2025-04-01", "2025-06-30"),
    ("2025-Q3", "2025-07-01", "2025-09-30"),
    ("2025-Q4", "2025-10-01", "2025-12-31"),
    ("2026-Q1", "2026-01-01", "2026-03-31"),
]

REGIME_TIMEOUT = TIME_BUDGET + 60


def annualize_return(total_return_pct: float, hours: int) -> float:
    if hours <= 0 or total_return_pct <= -100.0:
        return total_return_pct
    years = hours / 8760.0
    growth = 1.0 + total_return_pct / 100.0
    annual_growth = growth ** (1.0 / years)
    return (annual_growth - 1.0) * 100.0


def _run_window(args: tuple) -> dict:
    from strategy import Strategy

    name, start, end = args
    strategy = Strategy()
    data = load_data(start=start, end=end)

    total_bars = sum(len(df) for df in data.values())
    if total_bars == 0:
        return {"name": name, "start": start, "end": end, "error": "no data"}

    first_df = next(iter(data.values()))
    window_hours = len(first_df)

    result = run_backtest(strategy, data)
    base_score = compute_score(result)
    annual_return = annualize_return(result.total_return_pct, window_hours)

    return_gate = math.log(1.0 + max(annual_return, 0.0) / 100.0)
    score = base_score * return_gate if base_score > 0 else base_score

    return {
        "name": name,
        "start": start,
        "end": end,
        "bars": total_bars,
        "score": score,
        "sharpe": result.sharpe,
        "return_pct": result.total_return_pct,
        "annual_return_pct": annual_return,
        "max_dd_pct": result.max_drawdown_pct,
        "trades": result.num_trades,
        "win_rate": result.win_rate_pct,
        "profit_factor": result.profit_factor,
    }


def monthly_windows() -> list[tuple]:
    windows = []
    current = date(2025, 1, 1)
    end_cap = date(2026, 3, 31)
    while current <= end_cap:
        # month end
        if current.month == 12:
            month_end = date(current.year, 12, 31)
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
            month_end = next_month - timedelta(days=1)
        if month_end > end_cap:
            month_end = end_cap
        name = f"{current.year}-{current.month:02d}"
        windows.append((name, current.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        current = next_month
    return windows


def run_parallel(windows: list[tuple]) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=min(len(windows), 6)) as executor:
        futures = {executor.submit(_run_window, w): w for w in windows}
        for future in futures:
            name, start, end = futures[future]
            try:
                r = future.result(timeout=REGIME_TIMEOUT)
            except FuturesTimeout:
                r = {"name": name, "start": start, "end": end, "error": "timeout"}
            except Exception as e:
                r = {"name": name, "start": start, "end": end, "error": str(e)}
            results.append(r)
    order = {w[0]: i for i, w in enumerate(windows)}
    results.sort(key=lambda r: order.get(r["name"], 999))
    return results


def print_table(title: str, results: list[dict]) -> None:
    print(f"\n{'=' * 115}")
    print(title)
    print("-" * 115)
    print(f"{'Window':<10} {'Period':<25} {'Sharpe':>8} {'Ret%':>8} {'AnnRet%':>12} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Score':>7}")
    print("-" * 115)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<10} {r['start']}~{r['end']}  ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:<10} {r['start']}~{r['end']}"
            f"  {r['sharpe']:>8.2f}"
            f"  {r['return_pct']:>+7.1f}%"
            f"  {r['annual_return_pct']:>+11.1f}%"
            f"  {r['max_dd_pct']:>7.2f}%"
            f"  {r['trades']:>6}"
            f"  {r['win_rate']:>6.1f}%"
            f"  {r['profit_factor']:>5.1f}"
            f"  {r['score']:>6.2f}"
        )
    print("=" * 115)


def main():
    t_total = time.time()

    print("=== HOLDOUT SUB-PERIOD ANALYSIS ===")
    print("Purpose: identify which holdout sub-periods perform poorly (for diagnosis only, NOT optimization).\n")

    print("Running quarterly breakdown...")
    quarterly = run_parallel(QUARTERS)
    print_table("Quarterly breakdown (5 quarters)", quarterly)

    print("\nRunning monthly breakdown...")
    monthly = run_parallel(monthly_windows())
    print_table("Monthly breakdown", monthly)

    # Diagnostic summary
    valid_q = [r for r in quarterly if "error" not in r]
    valid_m = [r for r in monthly if "error" not in r]

    print("\n=== DIAGNOSTIC SUMMARY ===")
    if valid_q:
        worst_q = min(valid_q, key=lambda r: r["score"])
        print(f"Worst quarter:  {worst_q['name']}  Score={worst_q['score']:.2f}  DD={worst_q['max_dd_pct']:.2f}%  Sharpe={worst_q['sharpe']:.2f}")
        best_q = max(valid_q, key=lambda r: r["score"])
        print(f"Best quarter:   {best_q['name']}  Score={best_q['score']:.2f}  DD={best_q['max_dd_pct']:.2f}%  Sharpe={best_q['sharpe']:.2f}")
    if valid_m:
        worst_m = min(valid_m, key=lambda r: r["score"])
        print(f"Worst month:    {worst_m['name']}  Score={worst_m['score']:.2f}  DD={worst_m['max_dd_pct']:.2f}%  Return={worst_m['return_pct']:+.1f}%")
        worst3 = sorted(valid_m, key=lambda r: r["score"])[:3]
        print(f"Worst 3 months: " + ", ".join(f"{r['name']} ({r['score']:.1f})" for r in worst3))

        # DD analysis
        high_dd_months = [r for r in valid_m if r["max_dd_pct"] > 5.0]
        if high_dd_months:
            print(f"High-DD months (>5%):")
            for r in high_dd_months:
                print(f"  {r['name']}: DD={r['max_dd_pct']:.2f}%  Return={r['return_pct']:+.1f}%  Sharpe={r['sharpe']:.2f}")

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
