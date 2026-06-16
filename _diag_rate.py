"""Diagnostic: per-regime entry-rate distribution over trailing windows.
Measures how well 30 vs 100 vs 150-bar entry counts separate rally (churny)
from crash/sideways (rare-entry). Not committed; informs the grid gate."""
import numpy as np
from prepare import load_data, run_backtest
from strategy import Strategy, ACTIVE_SYMBOLS

REGIMES = [
    ("bull_2021", "2021-01-01", "2021-10-31"),
    ("crash_bear", "2021-11-01", "2022-12-31"),
    ("sideways", "2023-01-01", "2023-12-31"),
    ("rally_2024", "2024-01-01", "2024-12-31"),
]

# Monkeypatch Strategy to record every entry bar_count per symbol (unpruned).
_orig_init = Strategy.__init__
def _patched_init(self):
    _orig_init(self)
    self._diag_entries = {s: [] for s in ACTIVE_SYMBOLS}
Strategy.__init__ = _patched_init

for name, start, end in REGIMES:
    data = load_data(start=start, end=end)
    strat = Strategy()
    # Hook: wrap on_bar to snapshot entry history growth
    orig_on_bar = strat.on_bar
    def make_hook(s):
        def hook(bar_data, portfolio):
            sigs = orig_on_bar(bar_data, portfolio)
            for sym in ACTIVE_SYMBOLS:
                eh = s._entry_bar_history.get(sym, [])
                # record current full count is pruned to 30; instead track entries via _diag
            return sigs
        return hook
    # Simpler: run, then reconstruct entry bars from entry_bar_history is pruned.
    # Instead count entries via signals that open positions.
    res = run_backtest(strat, data)
    # Reconstruct: we need entry times. Use a fresh run capturing entries.
    print(f"{name}: trades={res.num_trades}")

# Second pass: capture entry bar_counts by instrumenting the entry append site indirectly.
print("\n--- trailing-window entry counts (per symbol, sampled) ---")
for name, start, end in REGIMES:
    data = load_data(start=start, end=end)
    strat = Strategy()
    entry_bars = {s: [] for s in ACTIVE_SYMBOLS}
    # Wrap: detect new entries by watching entry_bar dict changes
    prev_entry_bar = {s: None for s in ACTIVE_SYMBOLS}
    orig = strat.on_bar
    def wrapped(bar_data, portfolio, _s=strat, _eb=entry_bars, _pb=prev_entry_bar):
        sigs = orig(bar_data, portfolio)
        for sym in ACTIVE_SYMBOLS:
            cur = _s.entry_bar.get(sym, None)
            if cur is not None and cur != _pb[sym]:
                _eb[sym].append(cur)
            _pb[sym] = cur
        return sigs
    strat.on_bar = wrapped
    run_backtest(strat, data)
    # For each window W, compute distribution of trailing entry count at each entry bar
    for W in (30, 100, 150):
        counts = []
        for sym in ACTIVE_SYMBOLS:
            bars = sorted(entry_bars[sym])
            for b in bars:
                c = sum(1 for x in bars if b - W < x <= b)
                counts.append(c)
        if counts:
            counts = np.array(counts)
            print(f"{name:12s} W={W:3d}: mean={counts.mean():.2f} median={np.median(counts):.1f} "
                  f"p25={np.percentile(counts,25):.1f} p75={np.percentile(counts,75):.1f} max={counts.max()}")
    print()
