"""
Fetch recent klines from Hyperliquid API, append to local parquet cache,
then run backtest on the live trading period (2026-05-30 ~ 2026-06-01)
to compare with actual live trade results.

Usage: uv run scripts/replay_live_period.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare import load_data, run_backtest, DATA_DIR, SYMBOLS
from strategy import Strategy


def fetch_hl_candles(symbol: str, interval: str = "1h", start_ms: int = 0, end_ms: int = 0) -> pd.DataFrame:
    """Fetch candles from Hyperliquid public API."""
    url = "https://api.hyperliquid.xyz/info"
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms},
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for c in data:
        rows.append({
            "timestamp": int(c["t"]),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
            "funding_rate": 0.0,
        })
    return pd.DataFrame(rows)


def update_parquet_cache():
    """Extend parquet files with latest data from HL API."""
    for symbol in SYMBOLS:
        filepath = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
        existing = pd.read_parquet(filepath)
        last_ts = existing["timestamp"].iloc[-1]
        
        # Fetch from last timestamp to now
        end_ms = int(time.time() * 1000)
        start_ms = int(last_ts) + 3600_000  # next hour after last cached bar
        
        if start_ms >= end_ms:
            print(f"  {symbol}: already up to date")
            continue
        
        new_data = fetch_hl_candles(symbol, "1h", start_ms, end_ms)
        if len(new_data) == 0:
            print(f"  {symbol}: no new data")
            continue
        
        # Remove any overlap
        new_data = new_data[new_data["timestamp"] > last_ts]
        
        # Remove the last bar if it might be incomplete (current hour)
        current_hour_ms = int(pd.Timestamp.now(tz="UTC").floor("h").timestamp() * 1000)
        new_data = new_data[new_data["timestamp"] < current_hour_ms]
        
        if len(new_data) == 0:
            print(f"  {symbol}: no complete new bars")
            continue
        
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(filepath, index=False)
        
        new_last = pd.Timestamp(combined["timestamp"].iloc[-1], unit="ms", tz="UTC")
        print(f"  {symbol}: added {len(new_data)} bars, now up to {new_last}")


def main():
    print("=== STEP 1: Update parquet cache with latest HL data ===")
    update_parquet_cache()
    
    print("\n=== STEP 2: Run backtest on live trading period ===")
    # Live trading started 2026-05-30 16:00 UTC, we'll backtest the full period
    # Need warmup bars before, so start earlier
    data = load_data(start="2026-05-28", end="2026-06-02")
    
    if not data:
        print("ERROR: No data for the period. Check parquet cache.")
        return
    
    for sym, df in data.items():
        first = pd.Timestamp(df["timestamp"].iloc[0], unit="ms", tz="UTC")
        last = pd.Timestamp(df["timestamp"].iloc[-1], unit="ms", tz="UTC")
        print(f"  {sym}: {len(df)} bars ({first} to {last})")
    
    strategy = Strategy()
    result = run_backtest(strategy, data)
    
    print(f"\n=== BACKTEST RESULTS (2026-05-28 to 2026-06-02) ===")
    print(f"  Trades: {result.num_trades}")
    print(f"  Win rate: {result.win_rate_pct:.1f}%")
    print(f"  Sharpe: {result.sharpe:.2f}")
    print(f"  Return: {result.total_return_pct:+.3f}%")
    print(f"  MaxDD: {result.max_drawdown_pct:.3f}%")
    print(f"  Profit factor: {result.profit_factor:.2f}")
    
    print(f"\n=== COMPARISON WITH LIVE ===")
    print(f"  Live (2.5 days):   15 closed trades, WR=46.7%, PF=1.50, ret=+0.9%")
    print(f"  Backtest (same):   {result.num_trades} trades, WR={result.win_rate_pct:.1f}%, PF={result.profit_factor:.2f}, ret={result.total_return_pct:+.3f}%")
    print(f"\n  NOTE: Backtest uses 8bps taker fee + 1bps slippage (conservative).")
    print(f"  Live HL fee is ~0.5bps taker + ~0.6bps avg slippage = ~1.1bps total.")


if __name__ == "__main__":
    main()
