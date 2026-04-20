"""
Cumulative ablation: progressively remove HARMFUL components and track combined effect.
Each step builds on all previous removals.

Usage: uv run cumulative_ablation.py
"""

import re
import subprocess
import shutil
import time
import os

STRATEGY_FILE = "strategy.py"
BACKUP_FILE = "strategy.py.bak"


def run_backtest() -> dict:
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
    pattern = rf"^({param}\s*=\s*).*$"
    replacement = rf"\g<1>{new_value}"
    result = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if result == content:
        raise ValueError(f"Parameter {param} not found")
    return result


def remove_voter(content: str, bull_var: str, bear_var: str) -> str:
    def remove_var_from_sum_line(line: str, var: str) -> str:
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


def count_voters(content: str) -> int:
    """Count number of voters in bull_votes sum()."""
    for line in content.splitlines():
        if line.strip().startswith("bull_votes") and "sum(" in line:
            return line.count(",") + 1
    return 0


# Cumulative removal steps, ordered by single-ablation delta (most harmful first)
# Each step: (name, single_delta, modifications)
STEPS = [
    ("decel exit", +0.430,
     [("param", "DECEL_MULT_BASE", "999.0"), ("param", "DECEL_MULT_TREND", "999.0")]),
    ("EMA slope voter", +0.398,
     [("voter", "slope_bull", "slope_bear")]),
    ("EMA crossover voter", +0.361,
     [("voter", "ema_bull", "ema_bear")]),
    ("RSI voter", +0.196,
     [("voter", "rsi_bull", "rsi_bear")]),
    ("Vshort momentum voter", +0.117,
     [("voter", "vshort_bull", "vshort_bear")]),
    ("LinReg slope voter", +0.072,
     [("voter", "linreg_bull", "linreg_bear")]),
    ("vol_spike_scale", +0.068,
     [("param", "VOL_SPIKE_SCALE", "1.0")]),
    ("Momentum accel voter", +0.060,
     [("voter", "accel_bull", "accel_bear")]),
    ("Vol breakout voter", +0.034,
     [("voter", "vol_breakout_bull", "vol_breakout_bear")]),
    ("vol_compress_boost", -0.009,
     [("param", "VOL_COMPRESS_BOOST", "0.0")]),
]


def main():
    shutil.copy2(STRATEGY_FILE, BACKUP_FILE)
    original = open(STRATEGY_FILE).read()

    # Baseline
    print("Running baseline...")
    baseline = run_backtest()
    baseline_score = baseline.get("composite", -999.0)
    n_voters = count_voters(original)
    print(f"Baseline: {baseline_score:.6f} ({n_voters} voters)")
    print()

    cumulative = original
    prev_score = baseline_score
    results = []

    for i, (name, single_delta, mods) in enumerate(STEPS, 1):
        # Apply this step's modifications on top of all previous
        try:
            for mod in mods:
                if mod[0] == "param":
                    cumulative = replace_param(cumulative, mod[1], mod[2])
                elif mod[0] == "voter":
                    cumulative = remove_voter(cumulative, mod[1], mod[2])
        except ValueError as e:
            print(f"  SKIP step {i} ({name}): {e}")
            results.append((i, name, single_delta, None, None, None, "SKIP"))
            continue

        n_voters_now = count_voters(cumulative)

        # Check if MIN_VOTES needs adjustment
        # With fewer voters, the same MIN_VOTES becomes proportionally harder
        min_votes_note = ""
        if n_voters_now <= 4 and n_voters_now > 0:
            # Adjust MIN_VOTES to maintain ~30% threshold
            new_min = max(2, round(n_voters_now * 0.3))
            try:
                cumulative = replace_param(cumulative, "MIN_VOTES", str(new_min))
                min_votes_note = f" (MIN_VOTES -> {new_min})"
            except ValueError:
                pass

        with open(STRATEGY_FILE, "w") as f:
            f.write(cumulative)

        print(f"Step {i}: -{name} ({n_voters_now}v{min_votes_note})...", end=" ", flush=True)
        t0 = time.time()
        try:
            scores = run_backtest()
            score = scores.get("composite", -999.0)
            elapsed = time.time() - t0
            cum_delta = score - baseline_score
            step_delta = score - prev_score
            dd_max = max(scores.get("dd_bull_2021", 0), scores.get("dd_crash_bear", 0),
                         scores.get("dd_sideways", 0), scores.get("dd_rally_2024", 0))
            print(f"{score:.3f} (cum {cum_delta:+.3f}, step {step_delta:+.3f}, maxDD {dd_max:.1f}%) {elapsed:.0f}s")
            results.append((i, name, single_delta, score, cum_delta, step_delta, dd_max))
            prev_score = score
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((i, name, single_delta, None, None, None, "ERROR"))

    # Restore original
    shutil.copy2(BACKUP_FILE, STRATEGY_FILE)
    os.remove(BACKUP_FILE)

    # Summary
    print()
    print("=" * 110)
    print(f"{'Step':<5} {'Removed':<25} {'Single':>8} {'Score':>10} {'Cumul':>10} {'StepD':>10} {'MaxDD':>8}")
    print("-" * 110)
    print(f"{'0':<5} {'BASELINE':<25} {'':>8} {baseline_score:>10.3f} {'':>10} {'':>10}")

    best_score = baseline_score
    best_step = 0
    for i, name, single, score, cum, step, dd in results:
        if score is not None:
            dd_str = f"{dd:.1f}%" if isinstance(dd, float) else str(dd)
            print(f"{i:<5} {name:<25} {single:>+8.3f} {score:>10.3f} {cum:>+10.3f} {step:>+10.3f} {dd_str:>8}")
            if score > best_score:
                best_score = score
                best_step = i
        else:
            print(f"{i:<5} {name:<25} {single:>+8.3f} {'N/A':>10} {'N/A':>10} {'N/A':>10} {dd:>8}")

    print("=" * 110)
    print()
    print(f"Best cumulative score: {best_score:.3f} at step {best_step} (baseline: {baseline_score:.3f}, delta: {best_score - baseline_score:+.3f})")
    if best_step > 0:
        removed = [STEPS[j][0] for j in range(best_step)]
        print(f"Components removed: {', '.join(removed)}")
        kept_voters = count_voters(original)
        removed_voters = sum(1 for j in range(best_step) if STEPS[j][2][0][0] == "voter")
        print(f"Voters: {kept_voters} -> {kept_voters - removed_voters}")


if __name__ == "__main__":
    main()
