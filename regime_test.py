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

# Per-regime timeout: accounts for clean + len(FIXED_STABILITY_SEEDS)*N_TRIALS
# perturbed backtests (avg5 stability = 5 seeds x 20 trials = 100 perturbations,
# run STABILITY_WORKERS-way parallel). Measured 2026-06: full 4-regime parallel
# re-baseline run wall = 673s, which bounds the slowest single regime from above;
# 1020s = ~1.5x margin absorbs production CPU contention while still catching a
# pathological strategy (TIME_BUDGET=120s caps each backtest -> 100+ slow ones
# would far exceed this).
REGIME_TIMEOUT = TIME_BUDGET * 8 + 60


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

    # Score = base_score (already includes sharpe, dd_gate, vol_gate, streak_gate)
    score = base_score

    # Signal stability: penalize threshold-sensitive strategies
    from noise_test import compute_signal_stability, STABILITY_THRESHOLD
    if score > 0:
        stability = compute_signal_stability(data, result)
        # Continuous linear ramp: factor rises 0.0→1.0 as stability goes 0.50→0.80,
        # flat 1.0 at/above 0.80. Replaces the old 3-tier step function (×0.50/×0.75/×1.0),
        # which had cliffs at 0.70 and 0.80 — a 0.01 stability change could jump score 26%,
        # and within a tier improving stability gave zero factor gain (no gradient to track).
        # A continuous ramp gives a usable gradient at every stability value.
        STABILITY_FLOOR = 0.50
        stability_factor = min(
            1.0,
            max(0.0, (stability - STABILITY_FLOOR) / (STABILITY_THRESHOLD - STABILITY_FLOOR)),
        )
        score = score * stability_factor
    else:
        stability = 1.0
        stability_factor = 1.0

    # Flip streak gate: only apply when score is positive
    flip_streak_drag = result.flip_streak_total_drag  # <= 0
    if flip_streak_drag < 0 and score > 0:
        drag_per_bar = abs(flip_streak_drag) / max(total_bars, 1)
        flip_streak_gate = 1.0 / (1.0 + drag_per_bar / 0.5)
        score = score * flip_streak_gate
    else:
        flip_streak_gate = 1.0

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
        "stability": stability,
        "stability_factor": stability_factor,
        "flip_count": result.flip_count,
        "flip_win_rate": result.flip_win_rate_pct,
        "flip_pnl_pct": result.flip_total_pnl_pct,
        "flip_streak_drag": result.flip_streak_total_drag,
        "flip_streak_gate": flip_streak_gate,
    }


def compute_composite_score(results: list[dict]) -> float:
    """Composite = mean(scores) - k*std(scores). Returns -999 if any regime failed."""
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

        # Compute raw composite (before stability AND flip-streak penalties) for
        # keep threshold check. Must divide out BOTH gates that worker applied to
        # r["score"] (stability_factor at line ~85 and flip_streak_gate at ~95),
        # matching the per-regime raw_score reported below (which divides by
        # sf*fsg). Dividing by sf alone left a residual flip_streak_gate factor,
        # systematically deflating raw_composite when any regime had flip drag.
        raw_scores = []
        for r in results:
            if "error" not in r:
                sf = r.get("stability_factor", 1.0)
                fsg = r.get("flip_streak_gate", 1.0)
                denom = sf * fsg
                raw_s = r["score"] / denom if denom > 0 else r["score"]
                raw_scores.append(raw_s)
        if raw_scores:
            raw_mean = sum(raw_scores) / len(raw_scores)
            raw_var = sum((s - raw_mean) ** 2 for s in raw_scores) / len(raw_scores)
            raw_std = math.sqrt(raw_var)
            raw_composite = raw_mean - CONSISTENCY_K * raw_std
        else:
            raw_composite = -999.0

        # Parseable output for autoresearch agent
        print("---")
        print(f"composite_score:    {composite:.6f}")
        print(f"raw_composite:      {raw_composite:.6f}")
        print(f"mean_score:         {mean_s:.6f}")
        print(f"std_score:          {std_s:.6f}")
        print(f"num_regimes:        {len(scores)}")
        for r in results:
            if "error" not in r:
                n = r['name']
                sf = r.get('stability_factor', 1.0)
                fsg = r.get('flip_streak_gate', 1.0)
                raw = r['score'] / (sf * fsg) if (sf * fsg) > 0 else r['score']
                print(f"regime_{n}_score: {r['score']:.6f}")
                print(f"regime_{n}_raw_score: {raw:.6f}")
                print(f"regime_{n}_stability_factor: {sf:.6f}")
                print(f"regime_{n}_flip_streak_gate: {fsg:.6f}")
                print(f"regime_{n}_sharpe: {r['sharpe']:.6f}")
                print(f"regime_{n}_annual_return_pct: {r['annual_return_pct']:.6f}")
                print(f"regime_{n}_max_dd: {r['max_dd_pct']:.6f}")
                print(f"regime_{n}_stability: {r.get('stability', 1.0):.6f}")
                print(f"regime_{n}_flip_count: {r.get('flip_count', 0)}")
                print(f"regime_{n}_flip_wr: {r.get('flip_win_rate', 0.0):.2f}")
                print(f"regime_{n}_flip_pnl: {r.get('flip_pnl_pct', 0.0):.2f}")
                print(f"regime_{n}_flip_streak_drag: {r.get('flip_streak_drag', 0.0):.2f}")

        stabilities = [r.get("stability", 1.0) for r in results if "error" not in r]
        if stabilities:
            print(f"min_stability: {min(stabilities):.6f}")
        # Flip summary across all regimes
        total_flips = sum(r.get("flip_count", 0) for r in results if "error" not in r)
        total_trades = sum(r.get("trades", 0) for r in results if "error" not in r)
        flip_pnls_all = [r.get("flip_pnl_pct", 0.0) for r in results if "error" not in r]
        flip_wrs = [r.get("flip_win_rate", 0.0) for r in results if "error" not in r and r.get("flip_count", 0) > 0]
        print(f"total_flip_count: {total_flips}")
        print(f"flip_pct_of_trades: {total_flips / total_trades * 100:.1f}" if total_trades > 0 else "flip_pct_of_trades: 0.0")
        print(f"mean_flip_wr: {sum(flip_wrs) / len(flip_wrs):.2f}" if flip_wrs else "mean_flip_wr: 0.00")
