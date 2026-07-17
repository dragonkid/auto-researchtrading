#!/usr/bin/env python3
"""Compute owner-only promotion metrics for baseline vs candidate.

Dimensions:
- OKX development-source composite on the six development regimes
- sealed recent window (2026 Q2) on Binance
- unseen-token BNB Sharpe on the recent window

Detailed metrics are written only to the requested JSON path. Nothing is printed.
The caller is responsible for exposing only promotion_gate.py's sanitized result.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.cwd()))

import prepare
from prepare import compute_score, detect_warmup_bars, load_data, run_backtest

DEV_REGIMES = [
    ("2021-01-01", "2021-10-31"),
    ("2021-11-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31"),
]
RECENT = ("2026-04-01", "2026-06-12")


def load_strategy(commit: str):
    source = subprocess.run(
        ["git", "show", f"{commit}:strategy.py"], check=True,
        capture_output=True, text=True,
    ).stdout
    path = Path(tempfile.mkdtemp(prefix="promotion-strategy-")) / f"strategy_{commit}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"strategy_{commit}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strategy at {commit}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_window(strategy_mod, start: str, end: str):
    warm = detect_warmup_bars()
    warm_start = (pd.Timestamp(start) - pd.Timedelta(hours=warm)).strftime("%Y-%m-%d")
    data = load_data(start=warm_start, end=end)
    return run_backtest(strategy_mod.Strategy(), data, warmup_bars=warm)


def source_composite(strategy_mod, data_dir: str) -> float:
    prepare.DATA_DIR = data_dir
    prepare.SYMBOLS = ["BTC", "ETH", "SOL"]
    scores = [compute_score(run_window(strategy_mod, start, end)) for start, end in DEV_REGIMES]
    med = sorted(scores)[len(scores) // 2 - 1:len(scores) // 2 + 1]
    median = sum(med) / len(med)
    deviations = sorted(abs(s - median) for s in scores)
    mad = sum(deviations[2:4]) / 2
    return median - 0.5 * mad


def metrics_for(commit: str) -> dict:
    smod = load_strategy(commit)
    okx = source_composite(smod, os.path.expanduser("~/.cache/autotrader/data_okx"))

    prepare.DATA_DIR = os.path.expanduser("~/.cache/autotrader/data")
    prepare.SYMBOLS = ["BTC", "ETH", "SOL"]
    recent = run_window(smod, *RECENT)

    smod = load_strategy(commit)
    setattr(smod, "ACTIVE_SYMBOLS", ["BNB"])
    prepare.SYMBOLS = ["BNB"]
    unseen = run_window(smod, *RECENT)
    return {
        "okx": okx,
        "recent_sharpe": recent.sharpe,
        "recent_return": recent.total_return_pct,
        "recent_trades": recent.num_trades,
        "unseen": unseen.sharpe,
    }


def main() -> int:
    baseline, candidate, output = sys.argv[1:4]
    base = metrics_for(baseline)
    cand = metrics_for(candidate)
    payload = {
        "okx_delta": cand["okx"] - base["okx"],
        "recent_sharpe_delta": cand["recent_sharpe"] - base["recent_sharpe"],
        "recent_return_delta": cand["recent_return"] - base["recent_return"],
        "recent_trade_ratio": cand["recent_trades"] / max(base["recent_trades"], 1),
        "unseen_delta": cand["unseen"] - base["unseen"],
    }
    Path(output).write_text(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
