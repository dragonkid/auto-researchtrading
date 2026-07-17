#!/usr/bin/env python3
"""Measure baseline-vs-candidate changed completed-trade population.

Runs both commits on the six inspectable development regimes and compares only
PnL-realizing trade events. This is development data, so detailed output is safe
for the agent. The script uses a temporary worktree and restores nothing in the
caller's tree.
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
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REGIMES = [
    ("bull_2021", "2021-01-01", "2021-10-31"),
    ("crash_bear", "2021-11-01", "2022-12-31"),
    ("sideways", "2023-01-01", "2023-12-31"),
    ("rally_2024", "2024-01-01", "2024-12-31"),
    ("mixed_2025", "2025-01-01", "2025-12-31"),
    ("recent_2026q1", "2026-01-01", "2026-03-31"),
]


def event_key(row: tuple) -> tuple:
    # Kind, symbol, rounded executed price/PnL. Exclude delta size from identity so
    # pure sizing changes still count through changed PnL/price outcomes.
    return (row[0], row[1], round(float(row[3]), 6), round(float(row[4]), 6))


def load_strategy(commit: str):
    source = subprocess.run(
        ["git", "show", f"{commit}:strategy.py"], check=True,
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    path = Path(tempfile.mkdtemp(prefix="trade-pop-")) / f"strategy_{commit}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"strategy_{commit}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strategy at {commit}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_events(commit: str) -> dict[str, list[tuple]]:
    import prepare
    from internal_candidate import count_changed_trade_events

    smod = load_strategy(commit)
    warm = prepare.detect_warmup_bars()
    result: dict[str, list[tuple]] = {}
    for name, start, end in REGIMES:
        warm_start = (pd.Timestamp(start) - pd.Timedelta(hours=warm)).strftime("%Y-%m-%d")
        data = prepare.load_data(start=warm_start, end=end)
        run = prepare.run_backtest(smod.Strategy(), data, warmup_bars=warm)
        result[name] = [event_key(t) for t in run.trade_log if len(t) > 5 and t[5]]
    return result


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: uv run python scripts/trade_population_diff.py BASE CANDIDATE")
    base_commit, candidate_commit = sys.argv[1:]
    base = run_events(base_commit)
    candidate = run_events(candidate_commit)
    from internal_candidate import count_changed_trade_events
    per_regime = {
        name: count_changed_trade_events(base[name], candidate[name])
        for name, _, _ in REGIMES
    }
    print(json.dumps({
        "affected_trades": sum(per_regime.values()),
        "per_regime": per_regime,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
