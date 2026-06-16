"""
Signal stability via noise perturbation.

Simulates cross-exchange data-source differences using AR(1) correlated
noise matching empirical characteristics (Binance/HL vs CryptoCompare,
2 years of 1H data, 2024-01 ~ 2026-01).

Design: empirically calibrated stress test. STD values are the demeaned
cross-source random dispersion measured between the backtest data source
(cached CryptoCompare, ≈ Binance/OKX perp) and live venues. Constant basis
(spot-vs-perp ≈ 5bps) is excluded — it is a deterministic offset that any
change/relative signal is immune to; only demeaned random dispersion can flip
a signal near a decision boundary. A modest margin over the empirical median
is retained for robustness.

Anti-gaming: AC1 is a fixed grid across empirical range (not random, not fixed
single value), and seeds are derived from AST hash of strategy.py (immune to
comment/whitespace manipulation), preventing the autoresearch agent from
overfitting to specific noise characteristics or realizations.
"""

import ast
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import trim_mean

from prepare import run_backtest, BacktestResult

N_TRIALS = 20
STABILITY_THRESHOLD = 0.80  # no-penalty zone starts at 0.80
STABILITY_WORKERS = 4  # parallel trial workers inside one regime (nested under
# regime_test's 4-regime pool; total ~16 procs on a 14-core box, but only score>0
# regimes trigger stability so rarely all 4 run it at once). Parallelism does NOT
# change stability values: seeds are deterministic (base_seed+trial) and the
# trimmed mean is order-independent. Set to 1 to force serial.

# Empirical parameters: demeaned cross-source random std (basis removed).
# Measured 2026-06: cached source ≈ Binance/OKX perp; demeaned random dispersion
# (Binance↔OKX 2yr full sample + HL↔Binance/OKX recent) is close ~2.5-3.3 /
# high ~3.5-4.4 / low ~4-8 bps. Values below sit just above the empirical
# median, keeping a modest robustness margin without the prior 2-3x overstatement.
NOISE_CLOSE_STD_BPS = 3.0
NOISE_HIGH_STD_BPS = 4.0
NOISE_LOW_STD_BPS = 5.0

# AC1: fixed grid across empirical range (each trial = one difficulty level)
# All strategies face the same 20 AC1 values — enables paired comparison
NOISE_CLOSE_AC1_GRID = np.linspace(0.70, 0.97, N_TRIALS)
NOISE_HIGH_AC1_GRID = np.linspace(0.55, 0.83, N_TRIALS)
NOISE_LOW_AC1_GRID = np.linspace(0.29, 0.78, N_TRIALS)

CROSS_SYMBOL_CORR = 0.922  # sqrt(0.85) — achieves 0.85 cross-correlation

TRIM_FRACTION = 0.1  # 10% trimmed mean (drop 2 highest + 2 lowest from 20 trials)


def _strategy_hash():
    """Compute deterministic hash from strategy.py AST (immune to comments/whitespace)."""
    import os

    strategy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy.py")
    with open(strategy_path, "r") as f:
        tree = ast.parse(f.read())
    return int(hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16], 16)


def _generate_ar1(n, ac1, rng):
    """Generate AR(1) process, normalized to std=1."""
    innovation_std = np.sqrt(1.0 - ac1**2)
    series = np.empty(n)
    series[0] = rng.normal(0, 1)
    for t in range(1, n):
        series[t] = ac1 * series[t - 1] + innovation_std * rng.normal(0, 1)
    # Normalize to std=1 for consistent amplitude across trials
    std = series.std()
    if std > 1e-10:
        series /= std
    return series


def _perturb_data(data, rng, trial_idx):
    """Apply AR(1) correlated noise matching real cross-exchange differences.

    AC1 values come from fixed grid (trial_idx selects the difficulty level).
    """
    symbols = sorted(data.keys())  # deterministic order
    max_len = max(len(data[sym]) for sym in symbols)

    # AC1 from fixed grid — same difficulty for all strategies at this trial index
    close_ac1 = NOISE_CLOSE_AC1_GRID[trial_idx]
    high_ac1 = NOISE_HIGH_AC1_GRID[trial_idx]
    low_ac1 = NOISE_LOW_AC1_GRID[trial_idx]

    # Spawn child seeds for reproducibility (order-independent)
    ss = rng.bit_generator.seed_seq
    child_seeds = ss.spawn(3 + len(symbols) * 3)

    # Generate common drivers (one per price field, independent of each other)
    common_close = _generate_ar1(max_len, close_ac1, np.random.default_rng(child_seeds[0]))
    common_high = _generate_ar1(max_len, high_ac1, np.random.default_rng(child_seeds[1]))
    common_low = _generate_ar1(max_len, low_ac1, np.random.default_rng(child_seeds[2]))

    # Mixing weights: sqrt(ρ) for common, sqrt(1-ρ) for independent
    # Achieves cross-symbol correlation = ρ = CROSS_SYMBOL_CORR² ≈ 0.85
    common_weight = CROSS_SYMBOL_CORR  # sqrt(0.85) ≈ 0.922
    indep_weight = np.sqrt(1.0 - CROSS_SYMBOL_CORR**2)  # sqrt(0.15) ≈ 0.387

    result = {}
    for i, sym in enumerate(symbols):
        new_df = data[sym].copy()
        n = len(new_df)

        for field_idx, (field, common, ac1, std_bps) in enumerate(
            [
                ("close", common_close[:n], close_ac1, NOISE_CLOSE_STD_BPS),
                ("high", common_high[:n], high_ac1, NOISE_HIGH_STD_BPS),
                ("low", common_low[:n], low_ac1, NOISE_LOW_STD_BPS),
            ]
        ):
            seed_idx = 3 + i * 3 + field_idx
            indep_rng = np.random.default_rng(child_seeds[seed_idx])
            independent = _generate_ar1(n, ac1, indep_rng)
            noise = common_weight * common + indep_weight * independent
            noise_bps = noise * std_bps / 10000.0
            new_df[field] = new_df[field].values * (1.0 + noise_bps)

        # OHLC consistency
        new_df["high"] = np.maximum(new_df["high"].values, new_df["close"].values)
        new_df["low"] = np.minimum(new_df["low"].values, new_df["close"].values)
        new_df["high"] = np.maximum(new_df["high"].values, new_df["low"].values)

        result[sym] = new_df
    return result


# --- Parallel stability-trial machinery -------------------------------------
# Worker globals populated by _stability_init in each child process (avoids
# re-pickling the full OHLC dict per trial; only the trial index is mapped).
_W_DATA = None
_W_CLEAN_EQ = None
_W_CLEAN_RET = None
_W_CLEAN_VOL = None
_W_BASE_SEED = None


def _stability_init(data, clean_eq, clean_ret, clean_vol, base_seed):
    """ProcessPoolExecutor initializer: stash shared, read-only trial inputs."""
    global _W_DATA, _W_CLEAN_EQ, _W_CLEAN_RET, _W_CLEAN_VOL, _W_BASE_SEED
    _W_DATA = data
    _W_CLEAN_EQ = clean_eq
    _W_CLEAN_RET = clean_ret
    _W_CLEAN_VOL = clean_vol
    _W_BASE_SEED = base_seed


def _stability_trial(trial: int):
    """Run one perturbation trial -> tracking error (or None to skip).

    Pure function of (trial, worker-global inputs). Deterministic via
    base_seed+trial, so result is independent of execution order.
    """
    from strategy import Strategy

    rng = np.random.default_rng(_W_BASE_SEED + trial)
    perturbed_data = _perturb_data(_W_DATA, rng, trial)
    pert_result = run_backtest(Strategy(), perturbed_data)
    pert_eq = np.array(pert_result.equity_curve)

    if len(pert_eq) < 0.8 * len(_W_CLEAN_EQ):
        return 3.0 * _W_CLEAN_VOL

    n = min(len(_W_CLEAN_EQ), len(pert_eq))
    if n < 10:
        return None

    pert_ret = np.diff(pert_eq[:n]) / np.where(pert_eq[: n - 1] > 0, pert_eq[: n - 1], 1.0)
    diff = _W_CLEAN_RET[: n - 1] - pert_ret
    return float(diff.std())


def compute_signal_stability(data: dict, clean_result: BacktestResult) -> float:
    """Stability = 1 - normalized tracking error under AR(1) correlated noise.

    Seeding is derived from strategy.py AST hash to prevent overfitting to
    fixed noise realizations across iterations.
    """
    clean_eq = np.array(clean_result.equity_curve)
    if len(clean_eq) < 10:
        return 1.0

    clean_ret = np.diff(clean_eq) / np.where(clean_eq[:-1] > 0, clean_eq[:-1], 1.0)
    clean_vol = clean_ret.std()
    if clean_vol < 1e-10:
        return 1.0

    # Seed from strategy AST hash — same logic = same trials, different logic = different trials
    base_seed = _strategy_hash()

    init_args = (data, clean_eq, clean_ret, clean_vol, base_seed)
    raw = None
    if STABILITY_WORKERS > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=STABILITY_WORKERS,
                initializer=_stability_init,
                initargs=init_args,
            ) as ex:
                # map preserves order; trials are independent + deterministic so
                # the result is identical to the serial loop regardless of order.
                raw = list(ex.map(_stability_trial, range(N_TRIALS)))
        except Exception:
            raw = None  # nested-pool / spawn failure -> fall back to serial

    if raw is None:
        _stability_init(*init_args)
        raw = [_stability_trial(t) for t in range(N_TRIALS)]

    tracking_errors = [te for te in raw if te is not None]

    if not tracking_errors:
        return 1.0

    # 10% trimmed mean: robust to extreme favorable/unfavorable noise alignments
    te_array = np.array(tracking_errors)
    mean_te = float(trim_mean(te_array, proportiontocut=TRIM_FRACTION))
    normalized_te = mean_te / clean_vol
    return max(0.0, min(1.0, 1.0 - normalized_te))
