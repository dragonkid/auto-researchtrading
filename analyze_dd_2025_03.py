"""
Drill into 2025-03 to find the exact hours/days when DD spiked.

Warm up with 2025-01-01 to 2025-02-28 (~60 days) so the strategy has
realized state at the start of March, then track DD hour-by-hour
over 2025-03-01 to 2025-03-31.

Usage: uv run analyze_dd_2025_03.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from prepare import load_data, run_backtest, INITIAL_CAPITAL
from strategy import Strategy


WARMUP_START = "2025-01-01"
TARGET_START = "2025-03-01"
TARGET_END = "2025-03-31"


def main() -> None:
    data = load_data(start=WARMUP_START, end=TARGET_END)
    if not data:
        print("no data")
        return

    strategy = Strategy()
    result = run_backtest(strategy, data)
    eq = np.array(result.equity_curve)

    first_df = next(iter(data.values()))
    ts = pd.to_datetime(first_df["timestamp"], unit="ms", utc=True)
    n_eq = len(eq)
    n_bars = min(n_eq - 1, len(ts))
    records = []
    peak = eq[0]
    for i in range(1, n_bars + 1):
        peak = max(peak, eq[i])
        dd = (peak - eq[i]) / peak * 100.0
        records.append({
            "timestamp": ts.iloc[i - 1],
            "equity": eq[i],
            "peak": peak,
            "drawdown_pct": dd,
            "hourly_ret_pct": (eq[i] - eq[i - 1]) / eq[i - 1] * 100.0,
        })

    df = pd.DataFrame(records)
    df["date"] = df["timestamp"].dt.date

    target_start = pd.Timestamp(TARGET_START, tz="UTC")
    target_end = pd.Timestamp(TARGET_END + " 23:59", tz="UTC")
    march = df[(df["timestamp"] >= target_start) & (df["timestamp"] <= target_end)].copy()

    if march.empty:
        print("no March 2025 data available")
        return

    # Reset peak to only track March drawdowns
    march_peak = march["equity"].iloc[0]
    march_dd = []
    for eq_v in march["equity"]:
        march_peak = max(march_peak, eq_v)
        march_dd.append((march_peak - eq_v) / march_peak * 100.0)
    march["march_dd_pct"] = march_dd

    # Daily summary
    daily = march.groupby("date").agg(
        open_equity=("equity", "first"),
        close_equity=("equity", "last"),
        peak_equity=("equity", "max"),
        trough_equity=("equity", "min"),
        max_dd_pct_full=("drawdown_pct", "max"),
        max_dd_pct_march=("march_dd_pct", "max"),
    ).reset_index()
    daily["daily_return_pct"] = (daily["close_equity"] - daily["open_equity"]) / daily["open_equity"] * 100.0
    daily["intraday_swing_pct"] = (daily["peak_equity"] - daily["trough_equity"]) / daily["peak_equity"] * 100.0

    print("=" * 105)
    print("Daily breakdown for 2025-03")
    print("-" * 105)
    print(f"{'Date':<12} {'Open':>11} {'Close':>11} {'Daily%':>8} {'Swing%':>8} {'DD% (Mar)':>10} {'DD% (full)':>10}")
    print("-" * 105)
    for _, r in daily.iterrows():
        print(
            f"{str(r['date']):<12} {r['open_equity']:>11,.0f} {r['close_equity']:>11,.0f}"
            f"  {r['daily_return_pct']:>+7.2f}% {r['intraday_swing_pct']:>+7.2f}%"
            f"  {r['max_dd_pct_march']:>9.2f}% {r['max_dd_pct_full']:>9.2f}%"
        )
    print("=" * 105)

    # Show the full-run peak/trough and when they occurred
    idx_full_peak = march["equity"].idxmax()
    idx_full_dd = march["drawdown_pct"].idxmax()
    print(f"\nMarch peak equity:    {march['equity'].iloc[march.index.get_loc(idx_full_peak)]:,.0f}  at {march.loc[idx_full_peak, 'timestamp']}")
    print(f"March max DD (full):  {march.loc[idx_full_dd, 'drawdown_pct']:.2f}%  at {march.loc[idx_full_dd, 'timestamp']}")
    idx_m_dd = march["march_dd_pct"].idxmax()
    print(f"March max DD (March-only baseline): {march.loc[idx_m_dd, 'march_dd_pct']:.2f}% at {march.loc[idx_m_dd, 'timestamp']}")

    # Worst 10 hourly returns
    worst_hours = march.nsmallest(10, "hourly_ret_pct")[["timestamp", "hourly_ret_pct", "equity", "drawdown_pct"]]
    print("\nWorst 10 hourly returns in March:")
    print("-" * 70)
    for _, r in worst_hours.iterrows():
        print(f"  {r['timestamp']}  ret={r['hourly_ret_pct']:+7.2f}%  dd={r['drawdown_pct']:5.2f}%")

    # Check BTC/ETH/SOL price moves around the worst DD timestamp
    worst_ts = march.loc[idx_full_dd, "timestamp"]
    worst_ms = int(worst_ts.value // 1_000_000)
    print(f"\nAsset price moves ±12h around {worst_ts}:")
    for sym, df_sym in data.items():
        matches = df_sym.index[df_sym["timestamp"] == worst_ms]
        if len(matches) == 0:
            continue
        pos = matches[0]
        start = max(0, pos - 12)
        end = min(len(df_sym), pos + 12)
        seg = df_sym.iloc[start:end]
        move_24h = (seg["close"].iloc[-1] / seg["close"].iloc[0] - 1) * 100.0
        move_min = seg["close"].min() / seg["close"].iloc[0] * 100.0 - 100.0
        print(f"  {sym}: 24h price move {move_24h:+.2f}%  min-from-window-open {move_min:+.2f}%")


if __name__ == "__main__":
    main()
