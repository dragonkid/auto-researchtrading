"""
Signal stability via noise perturbation.

Compares clean vs perturbed equity curves to measure how sensitive
the strategy is to small data-source differences (±5 bps per bar).
"""

import numpy as np

from prepare import run_backtest, BacktestResult

N_TRIALS = 10
NOISE_BPS = 5.0
STABILITY_THRESHOLD = 0.85


def compute_signal_stability(data: dict, clean_result: BacktestResult) -> float:
    """
    Full-simulation stability via equity-curve tracking error.

    Returns: 1.0 (perfectly stable) to 0.0 (completely divergent).
    """
    from strategy import Strategy

    clean_eq = np.array(clean_result.equity_curve)
    if len(clean_eq) < 10:
        return 1.0

    clean_ret = np.diff(clean_eq) / np.where(clean_eq[:-1] > 0, clean_eq[:-1], 1.0)
    clean_vol = clean_ret.std()
    if clean_vol < 1e-10:
        return 1.0

    tracking_errors: list[tuple[bool, float]] = []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(42 + trial)
        correlated = trial < N_TRIALS // 2
        perturbed_data = _perturb_data(data, NOISE_BPS, rng, correlated=correlated)
        pert_result = run_backtest(Strategy(), perturbed_data)

        pert_eq = np.array(pert_result.equity_curve)

        if len(pert_eq) < 0.8 * len(clean_eq):
            tracking_errors.append((correlated, 3.0 * clean_vol))
            continue

        n = min(len(clean_eq), len(pert_eq))
        if n < 10:
            continue

        pert_ret = np.diff(pert_eq[:n]) / np.where(pert_eq[:n - 1] > 0, pert_eq[:n - 1], 1.0)
        clean_ret_aligned = clean_ret[: n - 1]

        diff = clean_ret_aligned - pert_ret
        te = diff.std()
        tracking_errors.append((correlated, te))

    if not tracking_errors:
        return 1.0

    corr_tes = [te for is_corr, te in tracking_errors if is_corr]
    iid_tes = [te for is_corr, te in tracking_errors if not is_corr]
    mean_te = max(
        sum(corr_tes) / len(corr_tes) if corr_tes else 0.0,
        sum(iid_tes) / len(iid_tes) if iid_tes else 0.0,
    )

    normalized_te = mean_te / clean_vol
    return max(0.0, min(1.0, 1.0 - normalized_te))


def _perturb_data(data: dict, noise_bps: float, rng, *, correlated: bool = False) -> dict:
    """Apply per-bar noise to close (and adjust H/L) for all symbols."""
    lengths = [len(df) for df in data.values()]
    if max(lengths) - min(lengths) > 5:
        raise ValueError(f"symbol bar count mismatch ({max(lengths) - min(lengths)}) too large for correlated noise")

    max_len = max(lengths)
    common_noise = rng.uniform(-noise_bps, noise_bps, size=max_len) / 10000.0 if correlated else None

    result = {}
    for sym, df in data.items():
        new_df = df.copy()
        n = len(new_df)
        noise = common_noise[:n] if correlated else rng.uniform(-noise_bps, noise_bps, size=n) / 10000.0
        new_df["close"] = new_df["close"] * (1.0 + noise)
        new_df["high"] = new_df[["high", "close"]].max(axis=1)
        new_df["low"] = new_df[["low", "close"]].min(axis=1)
        result[sym] = new_df
    return result
