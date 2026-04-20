"""
Ablation test: systematically disable each component and measure score impact.
Modifies strategy.py temporarily, runs regime_test.py, restores original.

Usage: uv run ablation_test.py
"""

import re
import subprocess
import shutil
import time

STRATEGY_FILE = "strategy.py"
BACKUP_FILE = "strategy.py.bak"


def run_backtest() -> dict:
    """Run regime_test.py and parse composite score + per-regime scores."""
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
            scores[f"regime_{name}"] = float(line.split(":")[1].strip())
        elif line.startswith("regime_") and "_max_dd:" in line:
            name = line.split("_max_dd:")[0].replace("regime_", "")
            scores[f"dd_{name}"] = float(line.split(":")[1].strip())
    return scores


def replace_param(content: str, param: str, new_value: str) -> str:
    """Replace a parameter value in strategy.py content."""
    pattern = rf"^({param}\s*=\s*).*$"
    replacement = rf"\g<1>{new_value}"
    result = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if result == content:
        raise ValueError(f"Parameter {param} not found in strategy.py")
    return result


def remove_from_votes(content: str, bull_var: str, bear_var: str) -> str:
    """Remove a voter from bull_votes and bear_votes sum lists only."""
    def remove_var_from_sum_line(line: str, var: str) -> str:
        # Remove ", var" or "var, " from inside sum([...])
        line = re.sub(rf",\s*{var}", "", line)
        line = re.sub(rf"{var}\s*,\s*", "", line)
        return line

    lines = content.splitlines(keepends=True)
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("bull_votes") and "sum(" in stripped:
            line = remove_var_from_sum_line(line, bull_var)
        elif stripped.startswith("bear_votes") and "sum(" in stripped:
            line = remove_var_from_sum_line(line, bear_var)
        result.append(line)
    return "".join(result)


# Ablation tests: (name, description, modifications)
# Each modification is either ("param", value) or ("remove_voter", bull_var, bear_var)
ABLATIONS = [
    # --- Voters ---
    ("Donchian voter", "Remove Donchian channel breakout voter",
     [("remove_voter", "donchian_bull", "donchian_bear")]),
    ("LinReg slope voter", "Remove linear regression slope voter",
     [("remove_voter", "linreg_bull", "linreg_bear")]),
    ("Momentum accel voter", "Remove momentum acceleration voter",
     [("remove_voter", "accel_bull", "accel_bear")]),
    ("Vol breakout voter", "Remove volatility breakout voter",
     [("remove_voter", "vol_breakout_bull", "vol_breakout_bear")]),
    ("EMA slope voter", "Remove EMA slope voter",
     [("remove_voter", "slope_bull", "slope_bear")]),
    ("MACD voter", "Remove MACD histogram voter",
     [("remove_voter", "macd_bull", "macd_bear")]),
    ("RSI voter", "Remove RSI direction voter",
     [("remove_voter", "rsi_bull", "rsi_bear")]),
    ("EMA crossover voter", "Remove EMA fast/slow crossover voter",
     [("remove_voter", "ema_bull", "ema_bear")]),
    ("Vshort momentum voter", "Remove very-short-term momentum voter",
     [("remove_voter", "vshort_bull", "vshort_bear")]),

    # --- Sizing multipliers ---
    ("calm_boost", "Disable calm regime position boost",
     [("param", "CALM_BOOST_MAX", "0.0")]),
    ("sideways_boost", "Disable sideways regime position boost",
     [("param", "SIDEWAYS_BOOST_MAX", "0.0")]),
    ("vol_spike_scale", "Disable vol-spike size reduction",
     [("param", "VOL_SPIKE_SCALE", "1.0")]),
    ("cross_asset_agree", "Disable cross-asset momentum agreement boost",
     [("param", "CROSS_ASSET_BOOST", "0.0")]),
    ("vote_boost", "Disable high-conviction vote sizing bonus",
     [("param", "HIGH_VOTE_BOOST", "0.0")]),
    ("vol_compress_boost", "Disable vol compression breakout boost",
     [("param", "VOL_COMPRESS_BOOST", "0.0")]),
    ("vol_confirm_mult", "Disable volume confirmation sizing",
     [("param", "VOL_CONFIRM_BOOST", "0.0"), ("param", "VOL_CONFIRM_FLOOR", "1.0")]),
    ("mtf_agree_mult", "Disable multi-timeframe agreement boost",
     [("param", "MTF_AGREE_BOOST", "0.0")]),
    ("strength_floor_sideways", "Reset sideways strength floor to default 0.6",
     [("param", "STRENGTH_FLOOR_SIDEWAYS", "0.6")]),

    # --- Entry/exit mechanisms ---
    ("mean-reversion entries", "Disable mean-reversion RSI entries in sideways",
     [("param", "MEANREV_TREND_THRESHOLD", "0.0")]),
    ("decel exit", "Disable momentum deceleration exit",
     [("param", "DECEL_MULT_BASE", "999.0"), ("param", "DECEL_MULT_TREND", "999.0")]),
    ("adaptive RSI period", "Use fixed RSI period (no sideways adaptation)",
     [("param", "RSI_PERIOD_SIDEWAYS", "8")]),
    ("adaptive cooldown", "Use fixed cooldown (no sideways reduction)",
     [("param", "COOLDOWN_SIDEWAYS_BARS", "2")]),
    ("vol-adaptive RSI exit", "Disable vol-adaptive RSI exit thresholds",
     [("param", "RSI_OB_TIGHT", "73"), ("param", "RSI_OS_TIGHT", "27")]),
    ("adaptive combined cap", "Use fixed combined mult cap (no vol adaptation)",
     [("param", "MAX_COMBINED_MULT_LOW_VOL", "5.5"), ("param", "MAX_COMBINED_MULT_HIGH_VOL", "5.5")]),
]


def classify_impact(delta: float) -> str:
    if delta <= -1.0:
        return "CRITICAL"
    elif delta <= -0.1:
        return "IMPORTANT"
    elif delta <= -0.01:
        return "MARGINAL"
    elif abs(delta) < 0.01:
        return "NEUTRAL"
    else:
        return "HARMFUL"


def main():
    # Backup original
    shutil.copy2(STRATEGY_FILE, BACKUP_FILE)
    original = open(STRATEGY_FILE).read()

    # Run baseline
    print("Running baseline...")
    t0 = time.time()
    baseline = run_backtest()
    baseline_time = time.time() - t0
    baseline_score = baseline.get("composite", -999.0)
    print(f"Baseline: {baseline_score:.6f} ({baseline_time:.1f}s)")
    print()

    results = []

    for name, desc, modifications in ABLATIONS:
        # Apply modifications
        modified = original
        try:
            for mod in modifications:
                if mod[0] == "param":
                    modified = replace_param(modified, mod[1], mod[2])
                elif mod[0] == "remove_voter":
                    modified = remove_from_votes(modified, mod[1], mod[2])
        except ValueError as e:
            print(f"  SKIP {name}: {e}")
            results.append((name, desc, None, None, "SKIP"))
            continue

        # Write modified strategy
        with open(STRATEGY_FILE, "w") as f:
            f.write(modified)

        # Run backtest
        print(f"Testing: {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            scores = run_backtest()
            score = scores.get("composite", -999.0)
            elapsed = time.time() - t0
            delta = score - baseline_score
            impact = classify_impact(delta)
            print(f"{score:.3f} ({delta:+.3f}) [{impact}] {elapsed:.0f}s")
            results.append((name, desc, score, delta, impact))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, desc, None, None, "ERROR"))

        # Restore original
        shutil.copy2(BACKUP_FILE, STRATEGY_FILE)

    # Clean up backup
    import os
    os.remove(BACKUP_FILE)

    # Summary table
    print()
    print("=" * 90)
    print(f"{'Component':<28} {'Score':>10} {'Delta':>10} {'Impact':<12} Description")
    print("-" * 90)
    print(f"{'BASELINE':<28} {baseline_score:>10.3f} {'':>10} {'':>12}")
    print("-" * 90)

    for name, desc, score, delta, impact in sorted(results, key=lambda x: x[3] if x[3] is not None else 999):
        if score is not None:
            print(f"{name:<28} {score:>10.3f} {delta:>+10.3f} {impact:<12} {desc}")
        else:
            print(f"{name:<28} {'N/A':>10} {'N/A':>10} {impact:<12} {desc}")

    print("=" * 90)

    # Recommendations
    print()
    critical = [r for r in results if r[4] == "CRITICAL"]
    important = [r for r in results if r[4] == "IMPORTANT"]
    marginal = [r for r in results if r[4] == "MARGINAL"]
    neutral = [r for r in results if r[4] == "NEUTRAL"]
    harmful = [r for r in results if r[4] == "HARMFUL"]

    if critical:
        print("KEEP (critical, >1.0 score drop):")
        for r in critical:
            print(f"  - {r[0]} ({r[3]:+.3f})")
    if important:
        print("KEEP (important, 0.1-1.0 drop):")
        for r in important:
            print(f"  - {r[0]} ({r[3]:+.3f})")
    if marginal:
        print("REVIEW (marginal, 0.01-0.1 drop):")
        for r in marginal:
            print(f"  - {r[0]} ({r[3]:+.3f})")
    if neutral:
        print("REMOVE (neutral, <0.01 impact):")
        for r in neutral:
            print(f"  - {r[0]} ({r[3]:+.3f})")
    if harmful:
        print("REMOVE (harmful, removing IMPROVES score):")
        for r in harmful:
            print(f"  - {r[0]} ({r[3]:+.3f})")


if __name__ == "__main__":
    main()
