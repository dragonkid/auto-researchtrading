"""
Ablation test on search regimes only (holdout preserved for final validation).

Reports per-regime impact for each component. Components that:
  - help all regimes → real alpha (KEEP)
  - help 1-2 regimes, hurt others → regime-fit (OVERFIT candidate)
  - negligible everywhere → dead code (REMOVE)

Usage: uv run ablation_test.py
"""

import re
import subprocess
import shutil
import time

STRATEGY_FILE = "strategy.py"
BACKUP_FILE = "strategy.py.bak"
REGIMES = ["bull_2021", "crash_bear", "sideways", "rally_2024"]


def run_backtest() -> dict:
    """Run regime_test.py and parse composite + per-regime scores."""
    result = subprocess.run(
        ["uv", "run", "regime_test.py"],
        capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    scores = {}
    for line in output.splitlines():
        if line.startswith("composite_score:"):
            scores["composite"] = float(line.split(":")[1].strip())
        elif line.startswith("mean_score:"):
            scores["mean"] = float(line.split(":")[1].strip())
        elif line.startswith("std_score:"):
            scores["std"] = float(line.split(":")[1].strip())
        elif line.startswith("regime_") and "_score:" in line:
            name = line.split("_score:")[0].replace("regime_", "")
            scores[name] = float(line.split(":")[1].strip())
    return scores


def replace_param(content: str, param: str, new_value: str) -> str:
    pattern = rf"^({param}\s*=\s*).*$"
    result = re.sub(pattern, rf"\g<1>{new_value}", content, count=1, flags=re.MULTILINE)
    if result == content:
        raise ValueError(f"Parameter {param} not found")
    return result


def remove_from_votes(content: str, bull_var: str, bear_var: str) -> str:
    """Remove voter variables only from the bull_votes/bear_votes sum() lines."""
    def scrub(line: str, var: str) -> str:
        line = re.sub(rf",\s*{var}\b", "", line)
        line = re.sub(rf"\b{var}\s*,\s*", "", line)
        return line

    out = []
    for line in content.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("bull_votes") and "sum(" in s:
            line = scrub(line, bull_var)
        elif s.startswith("bear_votes") and "sum(" in s:
            line = scrub(line, bear_var)
        out.append(line)
    return "".join(out)


# (name, description, [modifications])
ABLATIONS = [
    # --- Voters (9) ---
    ("voter: donchian", "Remove Donchian breakout voter",
     [("remove_voter", "donchian_bull", "donchian_bear")]),
    ("voter: linreg_slope", "Remove linear regression slope voter",
     [("remove_voter", "linreg_bull", "linreg_bear")]),
    ("voter: vol_breakout", "Remove volatility breakout voter",
     [("remove_voter", "vol_breakout_bull", "vol_breakout_bear")]),
    ("voter: ema_slope", "Remove EMA slope voter",
     [("remove_voter", "slope_bull", "slope_bear")]),
    ("voter: macd", "Remove MACD histogram voter",
     [("remove_voter", "macd_bull", "macd_bear")]),
    ("voter: rsi", "Remove RSI direction voter",
     [("remove_voter", "rsi_bull", "rsi_bear")]),
    ("voter: ema_crossover", "Remove EMA fast/slow crossover voter",
     [("remove_voter", "ema_bull", "ema_bear")]),
    ("voter: vshort_momentum", "Remove very-short-term momentum voter",
     [("remove_voter", "vshort_bull", "vshort_bear")]),
    ("voter: momentum", "Remove primary momentum voter",
     [("remove_voter", "mom_bull", "mom_bear")]),

    # --- Sizing multipliers ---
    ("size: calm_boost", "Disable calm-regime sizing boost",
     [("param", "CALM_BOOST_MAX", "0.0")]),
    ("size: sideways_boost", "Disable sideways-regime sizing boost",
     [("param", "SIDEWAYS_BOOST_MAX", "0.0")]),
    ("size: vol_compress_boost", "Disable vol-compression sizing boost",
     [("param", "VOL_COMPRESS_BOOST_MAX", "0.0")]),
    ("size: cross_asset_agree", "Disable cross-asset agreement boost",
     [("param", "CROSS_ASSET_BOOST", "0.0")]),
    ("size: vote_boost", "Disable high-conviction vote boost",
     [("param", "HIGH_VOTE_BOOST_MULT", "1.0")]),
    ("size: vol_confirm", "Disable volume confirmation sizing",
     [("param", "VOL_CONFIRM_FLOOR", "1.0"), ("param", "VOL_CONFIRM_CAP", "1.0")]),
    ("size: strength_floor_sw", "Reset sideways strength floor to default 0.6",
     [("param", "STRENGTH_FLOOR_SIDEWAYS", "0.6")]),
    ("size: trend_cap_boost", "Disable trend-adaptive combined-cap boost",
     [("param", "MAX_COMBINED_TREND_BOOST", "0.0")]),

    # --- Entry mechanisms ---
    ("entry: mean_reversion", "Disable RSI mean-reversion entries in sideways",
     [("param", "MEANREV_TREND_THRESHOLD", "0.0")]),
    ("entry: threshold_trend_red", "Disable trend-based threshold reduction",
     [("param", "TREND_THRESHOLD_SCALE", "0.0")]),
    ("entry: threshold_r2_red", "Disable R²-based threshold reduction",
     [("param", "LINREG_R2_THRESH_REDUCE", "0.0")]),
    ("entry: threshold_vol_compress", "Disable vol-compression threshold reduction",
     [("param", "VOL_COMPRESS_THRESH_REDUCE", "0.0")]),
    ("entry: trend_gate_deadzone", "Disable trend-gate deadzone bypass",
     [("param", "TREND_GATE_DEADZONE", "0.0")]),
    ("entry: r2_low_extra_vote", "Disable R²-low extra-vote requirement",
     [("param", "R2_LOW_THRESHOLD", "0.0")]),

    # --- Exit mechanisms ---
    ("exit: rsi_profit_tighten", "Disable profit-scaled RSI exit tightening",
     [("param", "RSI_EXIT_PROFIT_TIGHTEN", "0.0")]),
    ("exit: rsi_young_grace", "Disable young-position RSI exit grace period",
     [("param", "RSI_YOUNG_GRACE_BARS", "0")]),
    ("exit: peak_sideways_tighten", "Disable sideways peak-giveback tightening",
     [("param", "PEAK_PROFIT_SIDEWAYS_TIGHTEN", "0.0")]),
    ("exit: peak_age_loosen", "Disable age-based peak-giveback loosening",
     [("param", "PEAK_PROFIT_AGE_LOOSEN", "0.0")]),
    ("exit: peak_r2_loosen", "Disable R²-based peak-giveback loosening",
     [("param", "PEAK_PROFIT_R2_LOOSEN", "0.0")]),
]


def classify(deltas: dict) -> str:
    """Classify based on per-regime deltas to detect overfitting."""
    values = list(deltas.values())
    if not values:
        return "UNKNOWN"
    n_help = sum(1 for d in values if d < -0.05)   # regime got worse without this → component helps
    n_hurt = sum(1 for d in values if d > 0.05)    # regime got better without this → component hurts
    n_neutral = sum(1 for d in values if abs(d) <= 0.05)
    total_delta = sum(values)

    if n_neutral == len(values):
        return "DEAD"                   # no effect anywhere
    if n_help >= 3:
        return "ROBUST"                 # helps most regimes
    if n_help >= 1 and n_hurt >= 1:
        return "REGIME-FIT"             # helps some, hurts others → overfit candidate
    if n_hurt >= 2:
        return "HARMFUL"                # removing helps multiple regimes
    if total_delta > 0.1:
        return "HARMFUL"
    return "WEAK"


def main():
    shutil.copy2(STRATEGY_FILE, BACKUP_FILE)
    original = open(STRATEGY_FILE).read()

    try:
        print("Running baseline...")
        t0 = time.time()
        baseline = run_backtest()
        print(f"Baseline composite: {baseline.get('composite', 0):.3f}  "
              f"regimes: " + " ".join(f"{r}={baseline.get(r, 0):.2f}" for r in REGIMES)
              + f"  ({time.time()-t0:.0f}s)\n")

        results = []
        for name, desc, mods in ABLATIONS:
            modified = original
            try:
                for mod in mods:
                    if mod[0] == "param":
                        modified = replace_param(modified, mod[1], mod[2])
                    elif mod[0] == "remove_voter":
                        modified = remove_from_votes(modified, mod[1], mod[2])
            except ValueError as e:
                print(f"  SKIP {name}: {e}")
                results.append((name, desc, None, None, "SKIP"))
                continue

            with open(STRATEGY_FILE, "w") as f:
                f.write(modified)

            print(f"Testing: {name:<28}", end=" ", flush=True)
            t0 = time.time()
            try:
                scores = run_backtest()
                regime_deltas = {r: scores.get(r, 0) - baseline.get(r, 0) for r in REGIMES}
                composite_delta = scores.get("composite", 0) - baseline.get("composite", 0)
                label = classify(regime_deltas)
                deltas_str = " ".join(f"{d:+6.2f}" for d in regime_deltas.values())
                print(f"comp={composite_delta:+6.3f}  [{deltas_str}]  {label}  {time.time()-t0:.0f}s")
                results.append((name, desc, composite_delta, regime_deltas, label))
            except Exception as e:
                print(f"ERROR: {e}")
                results.append((name, desc, None, None, "ERROR"))

            shutil.copy2(BACKUP_FILE, STRATEGY_FILE)

        # Summary
        print("\n" + "=" * 110)
        print(f"{'Component':<28} {'Comp Δ':>8}  " + "  ".join(f"{r:>10}" for r in REGIMES) + "   Verdict")
        print("-" * 110)
        order = {"ROBUST": 0, "REGIME-FIT": 1, "WEAK": 2, "DEAD": 3, "HARMFUL": 4, "SKIP": 5, "ERROR": 6}
        for name, desc, comp_d, reg_d, label in sorted(results, key=lambda x: order.get(x[4], 99)):
            if comp_d is None:
                print(f"{name:<28} {'N/A':>8}  " + " " * 52 + f"   {label}")
                continue
            regime_str = "  ".join(f"{reg_d[r]:+10.2f}" for r in REGIMES)
            print(f"{name:<28} {comp_d:+8.3f}  {regime_str}   {label}")
        print("=" * 110)

        print("\nLegend:")
        print("  ROBUST     — removing hurts 3+ regimes → real alpha, keep")
        print("  REGIME-FIT — helps some regimes, hurts others → likely overfit")
        print("  HARMFUL    — removing improves multiple regimes → remove")
        print("  WEAK       — small mixed effects → low priority")
        print("  DEAD       — no effect anywhere → remove (LOC win)")

    finally:
        shutil.copy2(BACKUP_FILE, STRATEGY_FILE)
        import os
        os.remove(BACKUP_FILE)


if __name__ == "__main__":
    main()
