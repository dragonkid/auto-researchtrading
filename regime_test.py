"""
Regime robustness test: run the strategy across different market regimes.
Computes a composite score = mean(scores) - k*std(scores) to reward
strategies that work across ALL market conditions.

Usage: uv run regime_test.py
       uv run regime_test.py --holdout    # run holdout validation only
"""

import math
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET

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

# Per-regime timeout (seconds)
REGIME_TIMEOUT = TIME_BUDGET + 60


def annualize_return(total_return_pct: float, hours: int) -> float:
    """Convert total return to annualized return percentage."""
    if hours <= 0 or total_return_pct <= -100.0:
        return total_return_pct
    years = hours / 8760.0
    growth = 1.0 + total_return_pct / 100.0
    annual_growth = growth ** (1.0 / years)
    return (annual_growth - 1.0) * 100.0


def _run_regime_worker(args: tuple) -> dict:
    """Worker function for multiprocessing. Must be top-level for pickling."""
    from strategy import Strategy

    name, start, end, desc = args

    strategy = Strategy()
    data = load_data(start=start, end=end)

    total_bars = sum(len(df) for df in data.values())
    if total_bars == 0:
        return {"name": name, "desc": desc, "bars": 0, "error": "no data"}

    first_df = next(iter(data.values()))
    regime_hours = len(first_df)

    result = run_backtest(strategy, data)
    base_score = compute_score(result)
    annual_return = annualize_return(result.total_return_pct, regime_hours)

    # Return gate: penalize low absolute returns (prevents position-size gaming)
    # ann_return 1000% → gate 2.40, ann_return 100% → gate 0.69, ann_return 0% → gate 0
    return_gate = math.log(1.0 + max(annual_return, 0.0) / 100.0)
    score = base_score * return_gate if base_score > 0 else base_score

    return {
        "name": name,
        "desc": desc,
        "bars": total_bars,
        "score": score,
        "sharpe": result.sharpe,
        "return_pct": result.total_return_pct,
        "annual_return_pct": annual_return,
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

    use_holdout = "--holdout" in sys.argv
    regimes = HOLDOUT_REGIMES if use_holdout else SEARCH_REGIMES

    if use_holdout:
        print("=== HOLDOUT VALIDATION (not for autoresearch) ===\n")

    t_total = time.time()

    # Run all regimes in parallel
    results = []
    regime_order = {r[0]: i for i, r in enumerate(regimes)}

    print(f"Running {len(regimes)} regimes in parallel...\n")
    with ProcessPoolExecutor(max_workers=len(regimes)) as executor:
        futures = {executor.submit(_run_regime_worker, r): r for r in regimes}

        for future in futures:
            name, start, end, desc = futures[future]
            try:
                r = future.result(timeout=REGIME_TIMEOUT)
            except FuturesTimeout:
                r = {"name": name, "desc": desc, "bars": 0, "error": "timeout"}
            except Exception as e:
                r = {"name": name, "desc": desc, "bars": 0, "error": str(e)}
            results.append(r)

            if "error" in r:
                print(f"  {name}: ERROR — {r['error']}")
            else:
                print(f"  {name}: Sharpe={r['sharpe']:.2f}  AnnReturn={r['annual_return_pct']:+.1f}%  MaxDD={r['max_dd_pct']:.2f}%  Score={r['score']:.2f}")

    # Sort results back to original regime order
    results.sort(key=lambda r: regime_order.get(r["name"], 99))

    wall_time = time.time() - t_total
    print(f"\nTotal wall time: {wall_time:.1f}s")

    # Summary table
    print()
    print("=" * 120)
    print(f"{'Regime':<15} {'Period':<25} {'Sharpe':>8} {'AnnRet%':>10} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'PF':>6} {'Score':>8}")
    print("-" * 120)
    for (name, start, end, desc), r in zip(regimes, results):
        if "error" in r:
            print(f"{name:<15} {start}~{end}  {'ERROR':>8}  {r.get('error', '')}")
        else:
            print(
                f"{name:<15} {start}~{end}"
                f"  {r['sharpe']:>8.2f}"
                f"  {r['annual_return_pct']:>+9.1f}%"
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
                print(f"regime_{r['name']}_annual_return_pct: {r['annual_return_pct']:.6f}")
                print(f"regime_{r['name']}_max_dd: {r['max_dd_pct']:.6f}")
