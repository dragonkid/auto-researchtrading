# Statistical Deep-Dive: Monte Carlo Properties of the AR(1) Noise Test

## Executive Summary

The proposed 20-trial AR(1) noise test with randomized AC1 and strategy-hash seeding has
several statistical weaknesses that require attention:

1. **20 trials is marginally adequate** for the trimmed mean but leaves the standard error
   of the stability estimator at ~0.005-0.015 depending on the strategy, making the +0.003
   keep threshold potentially unresolvable.
2. **Unpaired comparisons inflate difference variance** by ~40-100% vs a paired design.
3. **AC1 randomization creates a heavy-tailed TE distribution** that 10% trimming may not
   fully tame.
4. **The +0.003 keep threshold is statistically marginal** under this design — it is likely
   within 1σ of the estimator noise for many strategy pairs.
5. **Concrete recommendations** are provided for improvement.

---

## 1. Effective Variance of Tracking Error Across Trials

### The Problem

With AC1 randomized uniformly from [0.70, 0.97], different trials produce fundamentally
different noise structures:

- AC1 = 0.70: half-life ≈ 2.0 bars, noise is nearly iid, innovations dominate
- AC1 = 0.97: half-life ≈ 23 bars, noise has persistent drift, wandering baseline

The tracking error (TE) is the std of (clean_ret - perturbed_ret). Under AR(1) noise:

  TE ∝ std_bps × f(strategy_sensitivity, AC1, n_bars)

The function f() varies enormously with AC1:
- Low AC1: perturbations are transient, strategies with any smoothing (EMA, lookback>3)
  are barely affected → low TE
- High AC1: perturbations persist for 20-30 bars, affecting all lookback-based indicators
  heavily → high TE

### Quantitative Estimate

For a typical momentum strategy with lookback L bars:
- Effective noise seen by the indicator ≈ std_bps × sqrt(half_life / L) for AC1 < 1
- At AC1=0.70 (HL≈2): indicator noise ≈ 7bps × sqrt(2/24) ≈ 2.0 bps
- At AC1=0.97 (HL≈23): indicator noise ≈ 7bps × sqrt(23/24) ≈ 6.9 bps

This is a 3.4× ratio in effective noise amplitude, translating to roughly a 3.4× ratio
in tracking error between the lowest-AC1 and highest-AC1 trials.

### Coefficient of Variation

With TE varying by ~3.4× across the AC1 range, the CV of the raw TE distribution across
20 trials is approximately:

  CV ≈ std(TE) / mean(TE) ≈ 0.4 - 0.6  (high variance)

For 20 samples from such a distribution, the standard error of the mean is:
  SE(mean) = std(TE) / sqrt(20) ≈ 0.10-0.13 × mean(TE)

For the 10%-trimmed mean (keeping 16 of 20 values):
  SE(trimmed_mean) ≈ std(middle 80%) / sqrt(16) × correction_factor
  ≈ 0.08-0.12 × mean(TE)

### Is 20 Trials Sufficient?

Marginally. The trimmed mean has SE ≈ 8-12% of the mean TE. After normalization by
clean_vol, if a typical stability value is 0.85 (i.e., normalized_TE = 0.15):

  SE(stability) ≈ 0.10 × 0.15 = 0.015

This means the 95% CI on a single stability measurement is approximately ±0.03.
**A +0.003 threshold is about 1/5 of a standard error** — essentially undetectable
at conventional significance levels from a single measurement.

However, if what we're doing is comparing Δstability = stability_new - stability_old
and both share the same noise realizations... except they don't (see Section 2).

---

## 2. Paired vs Unpaired: The Strategy-Hash Seeding Problem

### The Design Choice

The reform uses `base_seed = hash(strategy.py)` so that:
- Same code → same 20 noise paths (reproducible)
- Different code → different noise paths (anti-gaming)

### Statistical Consequence

When comparing strategy A (stability_A) vs strategy B (stability_B), the two measurements
use COMPLETELY DIFFERENT noise realizations. This means:

  Var(stability_A - stability_B) = Var(stability_A) + Var(stability_B)

In a PAIRED design (same noise for both), we would have:

  Var(stability_A - stability_B) = Var(stability_A) + Var(stability_B) - 2×Cov(A,B)

Since strategies that are minor variants of each other (typical in autoresearch) respond
similarly to the same noise, the correlation ρ(A,B) would be high (0.7-0.9), giving:

  Var_paired ≈ (1 - 0.8) × 2 × Var(single) = 0.4 × Var(single)
  Var_unpaired = 2 × Var(single)

The unpaired design has **5× higher variance** in stability differences compared to a
paired design for closely-related strategies.

### Impact on the +0.003 Threshold

Under the unpaired design:
  SE(Δstability) = sqrt(SE_A² + SE_B²) ≈ sqrt(2) × 0.015 ≈ 0.021

So a +0.003 improvement is only 0.003/0.021 ≈ 0.14 standard deviations — a z-score of
0.14, corresponding to p ≈ 0.44. **This is statistically meaningless.**

Even with a paired design:
  SE(Δstability)_paired ≈ sqrt(0.4) × 0.015 ≈ 0.009

A +0.003 improvement would be z ≈ 0.33, p ≈ 0.37. Still not significant.

### But Wait: Same Strategy Hash is Reproducible

There's a subtlety: the +0.003 threshold is applied to the SAME strategy re-evaluated
vs a baseline that was measured ONCE. If the baseline strategy doesn't change, its
stability number is deterministic (same hash → same trials → same result).

So the comparison is:
- baseline_stability: measured once, deterministic (no randomness given fixed code)
- new_stability: measured once, deterministic (different hash → different trials)

The issue is NOT sampling noise in individual measurements (those are deterministic given
the code hash). The issue is that different strategies face different noise regimes, so
stability is not on a common scale.

Strategy A might face 20 trials where, by chance:
- The AC1 draws tend to be lower (easier)
- The noise paths happen to align less with its signal structure

Strategy B might face 20 trials with:
- Higher AC1 draws (harder)
- Noise that happens to oppose its particular signal logic

### Quantifying the "Luck of the Draw"

For a given strategy, its hash determines:
- A fixed base_seed
- Which specific AC1 values are drawn (20 draws from U[0.70, 0.97])
- Which specific noise paths are generated

The mean of 20 draws from U[0.70, 0.97] has:
  E[mean AC1] = 0.835
  SE[mean AC1] = (0.97-0.70)/(sqrt(12)×sqrt(20)) = 0.27/(3.46×4.47) = 0.0175

So different strategies' mean AC1 across their 20 trials differs by about ±0.035 (2σ).
This translates to roughly ±5-10% difference in effective noise difficulty, which means
±0.005-0.015 in stability purely from luck of seed assignment.

**This is larger than the +0.003 threshold.**

---

## 3. Trimming Adequacy for the AC1-Induced Distribution Shape

### Distribution of TE Across Trials

With AC1 ~ U[0.70, 0.97]:
- TE is roughly proportional to a monotonically increasing function of AC1
- The distribution of TE is NOT symmetric — it's right-skewed because:
  - TE grows nonlinearly with AC1 (persistence compounds)
  - High-AC1 trials can produce aligned drift that either helps or hurts enormously

The TE distribution across 20 trials resembles:
- 7-8 trials with low TE (AC1 < 0.80, noise is transient)
- 7-8 trials with moderate TE (AC1 in 0.80-0.90)
- 4-5 trials with high TE (AC1 > 0.90, persistent drift)

### Is 10% Trimming Enough?

Trimming 10% removes 2 highest and 2 lowest from 20. The concern is:
- The 2-3 highest-AC1 trials (AC1 > 0.93) produce TE that may be 3-5× the median
- Only removing 2 of these leaves 1-3 extreme values in the trimmed set
- The remaining distribution still has a "bump" at the high end

A better approach might be:
- 20% trimming (remove 4 highest, 4 lowest) — provides breakdown point of 20%
- Median (50% breakdown) — more robust but less efficient
- Winsorized mean — clips extremes rather than removing them

### Recommendation

10% trimming has only a 10% breakdown point. Given the bimodal-ish structure (low-AC1
cluster vs high-AC1 cluster), I recommend:
- Either increase to 15-20% trimming (removing 3-4 from each tail)
- Or use the Hodges-Lehmann estimator (median of pairwise averages) which is
  highly efficient under contaminated distributions

---

## 4. Clean Vol as Denominator

### The Problem

The stability formula is:
  stability = 1 - trimmed_mean(TE) / clean_vol

Where clean_vol = std(clean equity returns) from a single backtest run.

Clean_vol is a SINGLE number computed from one deterministic backtest. It has no sampling
error in the Monte Carlo sense (it's deterministic given the data and strategy). So this
is actually fine from a variance-propagation standpoint.

### But There's a Different Problem

Clean_vol is the strategy's return volatility — which varies enormously across regimes.
The noise test runs per-regime, so each regime has its own clean_vol. This means:
- A high-vol regime (crash): clean_vol ≈ 0.02-0.05 → same TE gives lower stability
  Wait, no: higher clean_vol means TE/clean_vol is SMALLER → higher stability.
- A low-vol regime (sideways): clean_vol ≈ 0.005-0.01 → same TE gives LOWER stability

This means sideways/low-activity regimes will systematically have LOWER stability scores
because the denominator is small. This is a feature (strategies that barely trade are
sensitive to perturbation relative to their signal size) but should be understood.

### Estimation Error in Clean Vol (from finite bars)

Clean_vol is estimated from N bars of equity returns. For the standard deviation estimator:
  SE(clean_vol) / clean_vol ≈ 1/sqrt(2(N-1))

For typical regime sizes:
- 2000 bars: SE/vol ≈ 1.6% — negligible
- 500 bars: SE/vol ≈ 3.2% — small but present
- 200 bars: SE/vol ≈ 5.0% — non-trivial

Since this is deterministic (not Monte Carlo noise), it doesn't affect the variance
of the stability estimator across trials. But it does affect the absolute LEVEL of
stability — shorter regimes have slightly noisier denominators.

---

## 5. Better Statistical Frameworks

### Option A: Fixed AC1 Design with Paired Comparisons

Instead of randomizing AC1 per trial, fix it at a representative value (e.g., 0.93) and
use the SAME seed for all strategies. This enables:
- Paired difference tests
- Much lower variance of Δstability
- The +0.003 threshold becomes meaningful

Downside: loses the anti-gaming property. But you can retain anti-gaming by using a
COMMON set of trials across all strategies evaluated in the same session.

### Option B: Stratified AC1 Design

Divide the 20 trials into strata:
- Trials 1-7: AC1 drawn from [0.70, 0.80] (low persistence)
- Trials 8-14: AC1 drawn from [0.80, 0.90] (medium persistence)
- Trials 15-20: AC1 drawn from [0.90, 0.97] (high persistence)

Within each stratum, use the same seeds across strategies (enabling pairing).
Report stability per stratum and aggregate via weighted average.

Benefits:
- Ensures coverage of all AC1 regimes (current design could, by bad luck, draw
  15/20 trials from one end)
- Enables paired comparisons within strata
- More interpretable

### Option C: Bootstrap Confidence Intervals

Instead of the trimmed mean, use BCa bootstrap (2000 resamples of the 20 TE values)
to compute confidence intervals on stability. The keep threshold becomes:
- "Keep if lower bound of 90% CI on Δstability > 0"
This is more principled than a fixed +0.003.

But with only n=20 underlying observations, bootstrap CIs are notoriously unreliable
(coverage can be well below nominal). Need n≥50 for trustworthy bootstrap.

### Option D: Rank-Based Stability

Instead of comparing stability values, use a rank-based approach:
- Run both strategies on the SAME 20 noise realizations (paired)
- Count how many trials have TE_new < TE_old
- Keep if the win count ≥ 14/20 (roughly p<0.05, binomial test)

This is robust to distributional assumptions and heavy tails.

### Option E: Increase N_TRIALS to 50-100

The simplest fix: more trials. With n=50:
- SE(trimmed_mean) drops by sqrt(50/20) ≈ 1.6×
- The +0.003 threshold becomes ~0.5σ instead of ~0.2σ
- Still not "significant" but more reliable

Runtime concern: 20 trials already means 20 backtests per regime. Going to 50 would be
50 backtests × 4 regimes = 200 backtests per evaluation. If each takes ~5s, that's
~17 minutes. Probably too slow for autoresearch iteration.

### Recommended Framework: Paired Design with Stratified AC1

The best balance of statistical power, computational cost, and anti-gaming:

1. **Stratify AC1**: Pre-assign AC1 values for the 20 trials deterministically:
   AC1_values = linspace(0.70, 0.97, 20)  (or a Latin hypercube design)
   This eliminates the variance from AC1 randomization.

2. **Use strategy-independent seeding for the noise PATHS**: Seed the random innovations
   with fixed seeds (42+trial), but use strategy hash only for a small random permutation
   of which trial maps to which AC1 value. This preserves anti-gaming (strategy can't
   predict WHICH noise path has WHICH AC1) while ensuring:
   - All strategies face the same SET of (AC1, noise_path) combinations
   - Paired comparison is valid: Δstability computed trial-by-trial

3. **Paired keep threshold**: Instead of "stability improved by +0.003", use:
   "mean(TE_new - TE_old) < -0.003 × clean_vol" where the difference is computed
   trial-by-trial on matched noise paths.
   SE of this paired difference ≈ std(TE_diff) / sqrt(20)
   With pairing, std(TE_diff) ≈ 0.3 × std(TE) instead of sqrt(2) × std(TE)
   → SE drops from 0.021 to 0.005, making +0.003 a 0.6σ effect (still not great,
   but 4× better than the unpaired design).

---

## 6. Summary of Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| SE of stability estimator (~0.015) vs +0.003 threshold | HIGH | Threshold unresolvable |
| Unpaired design inflates Δstability variance by 5× | HIGH | Comparing strategies is noisy |
| AC1 randomization creates 3-5× TE range across trials | MEDIUM | Trimmed mean helps, not enough |
| Seed luck gives ±0.010 stability gift/penalty | HIGH | Larger than keep threshold |
| 10% trimming insufficient for skewed TE distribution | MEDIUM | Upward bias in stability |
| Clean_vol denominator estimation | LOW | Deterministic, no MC error |

### Key Conclusion

**The +0.003 keep threshold is NOT statistically meaningful under the proposed design.**

The "seed luck" effect (±0.010) and the estimator SE (~0.015) are both larger than the
threshold. A strategy could appear to gain +0.003 stability purely from receiving an
easier set of noise realizations (lower mean AC1 draws, favorable noise-signal alignment).

### Recommendations (in priority order)

1. **Make comparisons paired**: Use the same 20 noise paths for baseline and candidate.
   Derive seed from trial index, not strategy hash. Anti-gaming can be achieved by
   randomizing the AC1 assignment order via strategy hash.

2. **Stratify AC1**: Use fixed AC1 grid (not random draws) to eliminate AC1 lottery.
   Each strategy faces the exact same 20 difficulty levels.

3. **Increase threshold to +0.010** if keeping the unpaired design. This is closer to
   1σ of the difference estimator and provides weak but non-trivial discrimination.
   Alternatively, use a rank-based criterion (win 14+/20 trials).

4. **Consider 15-20% trimming** instead of 10% given the wide AC1 range.

5. **If computational budget allows, increase to 30-40 trials** for the stratified design.
   The Latin hypercube approach at n=30 gives SE ≈ 0.008, making +0.003 detectable at
   ~0.4σ under paired design.
