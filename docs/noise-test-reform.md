# Noise Test Reform: AR(1) Correlated Perturbation

## Background

The current `noise_test.py` uses iid uniform ±5bps perturbation on close/high/low per bar. Empirical analysis of cross-exchange price differences (Binance perps vs CryptoCompare spot, 2 years; HL perps vs CC, 5 months) reveals this model is fundamentally wrong:

- Real differences are **highly autocorrelated** (close AC1=0.92-0.97), not iid
- Real differences are **cross-symbol correlated** (0.82-0.93), not independent per symbol
- Real differences have **different characteristics per price field** (close: tight std + high AC; low: wide std + lower AC)

The consequence: iid noise tests produce stability rankings that anti-correlate with live performance. Empirically confirmed: binary voting (iid stability 0.8485) outperforms sigmoid voting (iid stability 0.8609) on HL holdout data (Sharpe 9.39 vs 8.65, MaxDD 7.30% vs 9.11%). The strategy optimized to score higher on the iid test performs worse in reality.

The fix: calibrate the noise model to match the empirical structure of real cross-exchange differences. Since real differences are persistent (AR(1) with AC1≈0.93), the corrected test will no longer reward strategies that merely resist per-bar random jitter while being slow to respond to sustained signals.

## Empirical Measurements

### Binance vs CryptoCompare (2 years, 2024-01 ~ 2026-01)

| Field | BTC std | ETH std | SOL std | BTC AC1 | ETH AC1 | SOL AC1 |
|-------|---------|---------|---------|---------|---------|---------|
| Close | 6.6 bps | 6.7 bps | 7.8 bps | 0.921 | 0.902 | 0.754 |
| High | 7.2 bps | 7.5 bps | 9.8 bps | 0.833 | 0.760 | 0.549 |
| Low | 8.5 bps | 9.4 bps | 19.2 bps | 0.637 | 0.553 | 0.287 |

### Recent 6 months (2025-12 ~ 2026-06)

| Field | BTC std | ETH std | SOL std | BTC AC1 | ETH AC1 | SOL AC1 |
|-------|---------|---------|---------|---------|---------|---------|
| Close | 5.2 bps | 5.0 bps | 5.2 bps | 0.968 | 0.956 | 0.924 |
| High | 5.6 bps | 6.5 bps | 5.7 bps | 0.805 | 0.635 | 0.711 |
| Low | 6.0 bps | 7.5 bps | 6.8 bps | 0.783 | 0.542 | 0.576 |

### Cross-symbol correlation (Binance-CC close diff)

- BTC-ETH: 0.930
- BTC-SOL: 0.818
- ETH-SOL: 0.841

### AC1 stability over time (BTC close, rolling 3-month windows)

AC1 ranges from 0.698 to 0.970 across 15 windows. 14/15 windows are >0.87. Only one anomaly (2024-09~2024-12: 0.698). Conclusion: high autocorrelation is a stable structural property.

### STD trend over time

STD has been declining (market efficiency improving):
- 2024 H1: 6.7-7.8 bps
- 2024 H2: 5.3-6.6 bps
- 2025 H1: 2.8-3.9 bps
- 2025 H2: 3.8-4.7 bps
- Recent 6mo: 5.0-5.2 bps

## Parameter Choice Rationale

**Design philosophy: conservative stress test.** Parameters are chosen to be more severe than current market conditions, ensuring strategies that pass will be robust even if market efficiency regresses to 2024 levels. This is analogous to using conservative fee assumptions in backtesting (8bps when real fees are 1bps).

**STD: use 2-year full sample (conservative/robust)**
- If strategy survives 7bps noise, it survives current 5bps trivially
- Covers possibility of market efficiency regression (2024-like conditions)
- Does not lose discriminating power (trading signals are 50-200bps, far above 7bps noise)

**AC1: use recent 6-month values (matches current market structure)**
- Recent AC1 is higher (0.93 vs 0.85 full sample), reflecting improved market microstructure
- Higher AC1 means real cross-exchange differences persist longer — the test should match this
- The combination (high STD + high AC1) is intentionally more severe than any single observed period, providing extra robustness margin

**Note on parameter mixing**: Using 2yr STD with 6mo AC1 creates a synthetic worst-case that was never observed empirically (when STD was 7bps in 2024, AC1 was ~0.85; when AC1 was 0.93 in 2025, STD was only 5bps). This is a deliberate conservative design choice, not an empirical claim.

**Final parameters:**

```python
NOISE_CLOSE_STD_BPS = 7.0    # 2yr conservative (actual range: 5.0-7.8)
NOISE_HIGH_STD_BPS = 8.0     # 2yr conservative (actual range: 5.6-9.8)
NOISE_LOW_STD_BPS = 12.0     # 2yr conservative (actual range: 6.0-19.2)
NOISE_CLOSE_AC1_RANGE = (0.70, 0.97)   # per-trial random (empirical range)
NOISE_HIGH_AC1_RANGE = (0.55, 0.83)    # per-trial random (empirical range)
NOISE_LOW_AC1_RANGE = (0.29, 0.78)     # per-trial random (empirical range)
CROSS_SYMBOL_CORR = 0.922    # sqrt(0.85) to achieve target 0.85 cross-correlation
N_TRIALS = 20                # all AR(1) correlated (no iid trials)
```

**Anti-gaming measures:**

1. **Per-trial AC1 randomization**: Each trial draws AC1 from the empirical range (uniform). This prevents the agent from tuning indicators to a specific decorrelation frequency. A strategy must be robust across the full range of observed autocorrelation structures.

2. **Strategy-deterministic seeding**: Instead of fixed seeds (42+trial), the master seed is derived from a hash of strategy.py content. This ensures:
   - Same strategy code → same 20 noise realizations (reproducible)
   - Different strategy code → different noise realizations (no overfitting to specific paths)
   - The agent cannot predict which noise paths it will face for a given modification

## Implementation

### Noise generation: AR(1) process

```
series[0] = N(0, 1)
series[t] = AC1 × series[t-1] + sqrt(1 - AC1²) × N(0, 1)
normalize series to std=1
```

### Cross-symbol correlation via common driver

The mixing formula `noise = a × common + b × independent` achieves cross-symbol correlation = a². To achieve target correlation ρ, set a = sqrt(ρ).

```
For each price field (close/high/low):
  1. Generate one common AR(1) driver (shared across symbols), normalized to std=1
  2. For each symbol, generate one independent AR(1) (same AC1), normalized to std=1
  3. Mix: noise = sqrt(ρ) × common + sqrt(1 - ρ) × independent
     (where ρ = 0.85, so sqrt(ρ) = 0.922, sqrt(1-ρ) = 0.387)
  4. Scale: noise_bps = noise × STD_BPS / 10000
  5. Apply: price *= (1 + noise_bps)
```

Resulting properties:
- Each symbol's noise: std ≈ STD_BPS, AC1 ≈ target AC1
- Cross-symbol correlation: sqrt(ρ)² = ρ = 0.85 ✓
- Noise variance: sqrt(ρ)² + sqrt(1-ρ)² = ρ + (1-ρ) = 1 ✓

### OHLC consistency (unchanged from current)

```
high = max(high, close)
low = min(low, close)
high = max(high, low)
```

### Reproducibility

- Each trial: `rng = np.random.default_rng(42 + trial)`
- Spawn child seeds via `rng.bit_generator.seed_seq.spawn()`
- 3 seeds for common drivers (close/high/low)
- 3 seeds per symbol for independent components
- Symbol processing order: `sorted(data.keys())` (deterministic)

### Stability computation

```
- 20 trials, all AR(1) correlated
- Aggregate TE using 10% trimmed mean (drop highest 2 + lowest 2, average remaining 16)
- normalized_te = trimmed_mean_te / clean_vol
- stability = clamp(1.0 - normalized_te, 0, 1)
```

Trimmed mean rationale: With AR(1) noise at AC1=0.93, some trials produce persistent drift aligned with strategy positions (artificially low TE) while others produce opposing drift (artificially high TE). Trimming removes these extremes, measuring the bulk behavior. Compared to plain mean (pulled by extremes) or 75th percentile (high variance at n=20, overly conservative), 10% trimmed mean provides near-optimal statistical efficiency (~98%) with good robustness (breakdown point 10%).

## Complete Implementation

```python
"""
Signal stability via noise perturbation.

Simulates cross-exchange data-source differences using AR(1) correlated
noise matching empirical characteristics (Binance/HL vs CryptoCompare,
2 years of 1H data, 2024-01 ~ 2026-01).

Design: conservative stress test. Parameters are intentionally more severe
than current market conditions to ensure robustness margin.

Anti-gaming: AC1 is randomized per trial (not fixed), and seeds are derived
from strategy code hash (not fixed constants), preventing the autoresearch
agent from overfitting to specific noise characteristics or realizations.
"""

import hashlib
import numpy as np
from scipy.stats import trim_mean
from prepare import run_backtest, BacktestResult

N_TRIALS = 20
STABILITY_THRESHOLD = 0.85  # MUST be recalibrated during verification step

# Empirical parameters (conservative worst-case design)
# STD: 2-year full sample (covers market efficiency regression)
NOISE_CLOSE_STD_BPS = 7.0
NOISE_HIGH_STD_BPS = 8.0
NOISE_LOW_STD_BPS = 12.0

# AC1: randomized per trial from empirical range (prevents frequency exploitation)
NOISE_CLOSE_AC1_RANGE = (0.70, 0.97)
NOISE_HIGH_AC1_RANGE = (0.55, 0.83)
NOISE_LOW_AC1_RANGE = (0.29, 0.78)

CROSS_SYMBOL_CORR = 0.922   # sqrt(0.85) — achieves 0.85 cross-correlation

TRIM_FRACTION = 0.1  # 10% trimmed mean (drop 2 highest + 2 lowest from 20 trials)


def _strategy_hash():
    """Compute a deterministic hash of strategy.py for seeding."""
    import importlib.util
    import os
    strategy_path = os.path.join(os.path.dirname(__file__), 'strategy.py')
    with open(strategy_path, 'rb') as f:
        return int(hashlib.sha256(f.read()).hexdigest()[:16], 16)


def _generate_ar1(n, ac1, rng):
    """Generate AR(1) process, normalized to std=1."""
    innovation_std = np.sqrt(1.0 - ac1 ** 2)
    series = np.empty(n)
    series[0] = rng.normal(0, 1)
    for t in range(1, n):
        series[t] = ac1 * series[t - 1] + innovation_std * rng.normal(0, 1)
    # Normalize to std=1 for consistent amplitude across trials
    std = series.std()
    if std > 1e-10:
        series /= std
    return series


def _perturb_data(data, rng):
    """Apply AR(1) correlated noise matching real cross-exchange differences.
    
    AC1 values are drawn randomly per call from empirical ranges.
    """
    symbols = sorted(data.keys())  # deterministic order
    max_len = max(len(data[sym]) for sym in symbols)

    # Draw AC1 for this trial from empirical ranges
    close_ac1 = rng.uniform(*NOISE_CLOSE_AC1_RANGE)
    high_ac1 = rng.uniform(*NOISE_HIGH_AC1_RANGE)
    low_ac1 = rng.uniform(*NOISE_LOW_AC1_RANGE)

    # Spawn child seeds for reproducibility (order-independent)
    ss = rng.bit_generator.seed_seq
    child_seeds = ss.spawn(3 + len(symbols) * 3)

    # Generate common drivers (one per price field, independent of each other)
    common_close = _generate_ar1(max_len, close_ac1, np.random.default_rng(child_seeds[0]))
    common_high = _generate_ar1(max_len, high_ac1, np.random.default_rng(child_seeds[1]))
    common_low = _generate_ar1(max_len, low_ac1, np.random.default_rng(child_seeds[2]))

    # Mixing weights: sqrt(ρ) for common, sqrt(1-ρ) for independent
    # Achieves cross-symbol correlation = ρ = CROSS_SYMBOL_CORR² ≈ 0.85
    common_weight = CROSS_SYMBOL_CORR        # sqrt(0.85) ≈ 0.922
    indep_weight = np.sqrt(1.0 - CROSS_SYMBOL_CORR ** 2)  # sqrt(0.15) ≈ 0.387

    result = {}
    for i, sym in enumerate(symbols):
        new_df = data[sym].copy()
        n = len(new_df)

        for field_idx, (field, common, ac1, std_bps) in enumerate([
            ('close', common_close[:n], close_ac1, NOISE_CLOSE_STD_BPS),
            ('high', common_high[:n], high_ac1, NOISE_HIGH_STD_BPS),
            ('low', common_low[:n], low_ac1, NOISE_LOW_STD_BPS),
        ]):
            seed_idx = 3 + i * 3 + field_idx
            indep_rng = np.random.default_rng(child_seeds[seed_idx])
            independent = _generate_ar1(n, ac1, indep_rng)
            noise = common_weight * common + indep_weight * independent
            noise_bps = noise * std_bps / 10000.0
            new_df[field] = new_df[field].values * (1.0 + noise_bps)

        # OHLC consistency
        new_df['high'] = np.maximum(new_df['high'].values, new_df['close'].values)
        new_df['low'] = np.minimum(new_df['low'].values, new_df['close'].values)
        new_df['high'] = np.maximum(new_df['high'].values, new_df['low'].values)

        result[sym] = new_df
    return result


def compute_signal_stability(data, clean_result):
    """Stability = 1 - normalized tracking error under AR(1) correlated noise.
    
    Seeding is derived from strategy.py hash to prevent overfitting to
    fixed noise realizations across iterations.
    """
    from strategy import Strategy

    clean_eq = np.array(clean_result.equity_curve)
    if len(clean_eq) < 10:
        return 1.0

    clean_ret = np.diff(clean_eq) / np.where(clean_eq[:-1] > 0, clean_eq[:-1], 1.0)
    clean_vol = clean_ret.std()
    if clean_vol < 1e-10:
        return 1.0

    # Seed from strategy code hash — same code = same trials, different code = different trials
    base_seed = _strategy_hash()
    tracking_errors = []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(base_seed + trial)
        perturbed_data = _perturb_data(data, rng)
        pert_result = run_backtest(Strategy(), perturbed_data)
        pert_eq = np.array(pert_result.equity_curve)

        if len(pert_eq) < 0.8 * len(clean_eq):
            tracking_errors.append(3.0 * clean_vol)
            continue

        n = min(len(clean_eq), len(pert_eq))
        if n < 10:
            continue

        pert_ret = np.diff(pert_eq[:n]) / np.where(pert_eq[:n-1] > 0, pert_eq[:n-1], 1.0)
        diff = clean_ret[:n-1] - pert_ret
        tracking_errors.append(diff.std())

    if not tracking_errors:
        return 1.0

    # 10% trimmed mean: robust to extreme favorable/unfavorable noise alignments
    te_array = np.array(tracking_errors)
    mean_te = float(trim_mean(te_array, proportiontocut=TRIM_FRACTION))
    normalized_te = mean_te / clean_vol
    return max(0.0, min(1.0, 1.0 - normalized_te))
```

## Verification Plan

### Step 1: Noise property validation (unit test)

Before running on strategies, verify the noise generation produces correct statistical properties:
- Generate 100 realizations, measure empirical AC1, std, and cross-symbol correlation
- Confirm they match targets within confidence intervals
- Plot example noise paths and compare visually with real CC-vs-HL difference series

### Step 2: Multi-strategy comparison

Run the new noise test on 5+ strategies with known different characteristics:
- 8569cb5 (binary voting, live Sharpe 9.39)
- c614b8b (sigmoid voting, live Sharpe 8.65)
- bd3f399 (confidence sizing)
- 8569cb5 with artificially widened thresholds (known worse)
- A trivially stable strategy (buy-and-hold or always-flat) as calibration anchor

### Step 3: Verify ranking alignment

- Confirm binary > sigmoid under new noise test (matching live performance ranking)
- Compute confidence intervals on stability differences (are they statistically significant?)
- Check that trivially stable strategies score near 1.0 (sanity check)

### Step 3b: Sensitivity analysis

Since AC1 is now randomized per trial, the sensitivity analysis shifts focus:
- Run the noise test with AC1 ranges narrowed to specific bands: [0.70-0.75], [0.80-0.85], [0.90-0.97]
- Verify that binary > sigmoid holds across ALL bands (not just the high-AC1 range)
- If the ranking only holds for high-AC1 bands, the reform's conclusions are fragile
- If it holds across all bands, the conclusion is robust to the specific AC1 distribution

This validates that the reform's benefits come from using correlated (non-iid) noise in general, not from a specific AC1 value.

### Step 4: Threshold calibration

- Record stability values for all tested strategies
- Set STABILITY_THRESHOLD such that the known-good production strategy (8569cb5) falls in the "no penalty" zone (≥0.90)
- Adjust penalty tiers in regime_test.py accordingly
- This MUST happen before resuming autoresearch

### Step 5: Deploy

- Replace noise_test.py
- Update regime_test.py thresholds
- Update program-stateless.md keep threshold if needed
- Restart autoresearch

## Risks

1. **STABILITY_THRESHOLD will need recalibration** — new model has larger noise amplitude, absolute stability values will decrease. Calibration is part of verification (Step 4), not deferred.
2. **Keep threshold +0.003 may need adjustment** — different noise model = different sensitivity per unit of architectural change. Determine empirically during verification.
3. **Entire stability trajectory (0.73→0.86) is invalidated** — those numbers were measured under iid; they have no meaning under correlated noise.
4. **Agent exploration history was shaped by iid** — many "confirmed dead ends" may not be dead under correlated noise (e.g., binary voting mechanisms that were "too noisy" under iid might be perfectly stable under correlated).
5. **Per-symbol differences (especially SOL low: 19.2bps std, AC1=0.29)** are simplified to single parameters. If SOL-specific behavior is critical, consider per-symbol params in a future iteration.

## Changes from Current Implementation

| Aspect | Old | New |
|--------|-----|-----|
| Noise model | iid uniform ±5bps | AR(1) Gaussian, per-field params |
| Time autocorrelation | None (each bar independent) | Per-trial random: close [0.70-0.97], high [0.55-0.83], low [0.29-0.78] |
| Cross-symbol | shared uniform noise (correlated trials) | sqrt(0.85)≈0.922 loading, achieving ρ=0.85 |
| Amplitude | uniform ±5bps (std≈2.9bps) | Gaussian std 7/8/12 bps |
| Distribution | Uniform | Gaussian (AR(1) innovations) |
| Trial types | 10 correlated + 10 iid | 20 all AR(1) correlated |
| TE aggregation | max(corr_mean, iid_mean) | 10% trimmed mean |
| Seeding | Fixed (42+trial) | Strategy-code-hash derived (anti-gaming) |
| AC1 | N/A (iid) | Randomized per trial from empirical range (anti-gaming) |
| Parameter design | Single fixed value | Conservative worst-case STD + randomized AC1 |

## Deployment: New Branch

Deploy as `autotrader/score-v4-run1` branch to cleanly separate from the iid-optimized history in `score-v3-run1`. The agent starts fresh with no prior results.tsv baggage — all "confirmed dead ends" from iid optimization are irrelevant under the new noise model.
