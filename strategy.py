import numpy as np
from scipy.stats import linregress
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]

# Momentum windows
MED_WINDOW_MIN = 8
MED_WINDOW_MAX = 16
MED2_WINDOW = 10
SHORT_WINDOW = 8
LONG_WINDOW = 20

# EMA parameters
EMA_FAST = 3
EMA_SLOW = 21
EMA_SLOPE_PERIOD = 22
EMA_SLOPE_LOOKBACK = 3

# MACD parameters
MACD_FAST = 8
MACD_SLOW = 16
MACD_SIGNAL = 8  # widened from 4->6->7->8 to smooth MACD histogram further

# Linear regression
LINREG_PERIOD = 16

# Volatility parameters
VOL_LOOKBACK = 24
VOL_SHORT_LOOKBACK = 12
VOL_LONG_LOOKBACK = 36
TARGET_VOL = 0.015

# Entry threshold
BASE_THRESHOLD = 0.005
DYN_THRESHOLD_FLOOR = 0.00475
DYN_THRESHOLD_CEIL = 0.012
TREND_THRESHOLD_SCALE = 0.25       # max threshold reduction in trends (reduced from 0.32 for wider buffer in rally)
TREND_THRESHOLD_DECAY = 0.14       # abs(ret_long) at which reduction saturates

# RSI voter
RSI_TREND_BIAS = 2.0
RSI_TREND_BIAS_DECAY = 0.10

# Exit parameters (momentum-decay + slope + peak-profit + stop-loss)
HOLD_DECAY_START = 6   # bars after which exit pressure begins
HOLD_DECAY_RATE = 0.25  # exit pressure per bar beyond start (0.25 = exit at bar 10 with no momentum)
MOMENTUM_HOLD_BONUS = 2  # max extra bars when slope strongly agrees (conservative cap)
STOP_LOSS_PCT = -0.024
PEAK_PROFIT_MIN_BASE = 0.025
PEAK_PROFIT_GIVEBACK = 0.22

# Sizing multipliers
BASE_POSITION_SIZE = 0.065
CALM_BOOST_MAX = 0.8
SIDEWAYS_BOOST_MAX = 0.50
CROSS_ASSET_FIXED_BOOST = 0.15
HIGH_VOTE_BOOST_MULT = 1.20
VOL_CONFIRM_LOOKBACK = 12
VOL_CONFIRM_BASE = 24
VOL_CONFIRM_FLOOR = 0.98
VOL_CONFIRM_CAP = 1.10
STRENGTH_FLOOR_SIDEWAYS = 2.6
STRENGTH_FLOOR_DECAY = 0.12

# Combined mult cap
MAX_COMBINED_MULT_HIGH_VOL = 2.5
MAX_COMBINED_MULT_LOW_VOL = 6.5
MAX_COMBINED_VOL_LOW = 0.6
MAX_COMBINED_VOL_HIGH = 1.2
MAX_COMBINED_TREND_BOOST = 1.0

# Trend gate
TREND_GATE_MED_WEIGHT_SIDEWAYS = 0.85
TREND_GATE_MED_WEIGHT_BASE = 0.70
TREND_GATE_DEADZONE = 0.018
# Vote / cooldown (6 voters, soft tanh contributions)
# Strong-consensus weighted sum: replaces hard count of voters above STRONG_CONF
# with sum of (conf-0.5)*2 for conf>0.5, weighted by margin. Removes noise boundary at 0.65.
STRONG_WEIGHT_MIN = 1.5  # required sum of margin-above-0.5 voter contributions
MIN_VOTES = 2.5
FLIP_MIN_VOTES = 2.4  # slightly looser to admit protective flips in rally
COOLDOWN_BARS = 1
COOLDOWN_TREND_DECAY = 0.06


def ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

# Position accumulation (build position over bars)
# Architectural: vol-conditioned initial commit. Base 0.43, modulated by vol_ratio
# via tanh: low vol -> larger initial frac (~0.50, capture rally/sideways momentum
# with less first-bar lag); high vol -> smaller initial frac (~0.36, less DD on
# wrong-side first bar in crash). Continuous, bounded ~[0.36, 0.50].
ENTRY_INITIAL_FRAC_BASE = 0.43
ENTRY_INITIAL_FRAC_VOL_AMP = 0.07
ENTRY_INITIAL_FRAC = 0.43  # retained for scale-in start anchor + flip-fraction path
ENTRY_FULL_BARS = 3  # bars to reach full position (linear scale-in over 3 bars)


class Strategy:
    def __init__(self):
        self.entry_prices, self.exit_bar, self.peak_pnl, self.entry_bar = {}, {}, {}, {}
        self.bar_count = 0
        self.smoothed_trend = {}
        # Two prior pnl bars for confirmed-peak gate (need 2 rising bars to update).
        self._smoothed_pnl = {}
        # Persistence buffers: last 2 bars of strong-side firings per symbol.
        # Used to TIGHTEN _strong_min on isolated single-bar firing spikes (noise filter).
        self._bull_strong_hist = {}
        self._bear_strong_hist = {}
        # Architectural: flip-origin tracker. True when current position originated
        # from a flip (high-conviction reversal: both vote count AND trend sign +
        # opposite-side strong-min admission). Used in exit logic to give flips
        # extra maturation time before the exit-pressure gate can fire.
        self._from_flip = {}

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        # Architectural: portfolio-level concentration attenuator on cold-entry size.
        # Cross-symbol data dependency at the sizing stage. Compute net same-side
        # exposure: count of active positions sharing the candidate side. Attenuates
        # first-bar size when 2+ symbols already hold same-side positions (high
        # correlation risk). Applied only on cold entry path (not flips, which are
        # protective and need full size). Continuous via tanh of count / scale; one-sided.
        # Long count and short count are computed pre-loop so each symbol sees the
        # state from PRIOR bar's positions (no in-bar feedback loops).
        _long_count_pre = sum(1 for s in ACTIVE_SYMBOLS if portfolio.positions.get(s, 0.0) > 0)
        _short_count_pre = sum(1 for s in ACTIVE_SYMBOLS if portfolio.positions.get(s, 0.0) < 0)

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close
            realized_vol = max(np.std(np.diff(np.log(closes[-VOL_LOOKBACK - 1:-1]))), 1e-6)
            # Architectural: per-symbol adaptive vol baseline. Replace constant TARGET_VOL
            # with long-window (200-bar) realized vol blended with TARGET_VOL anchor at
            # 0.5 weight. Long-window vol is each symbol's structural baseline (BTC ~0.012,
            # SOL ~0.022); blending with anchor preserves global scaling but reduces SOL's
            # bias toward "always-elevated" vol_ratio. New cross-bar data dependency on
            # 200-bar log-return std per symbol; smooth (no boundary), continuous.
            _long_n = min(200, len(closes) - 1)
            _baseline_vol = max(np.std(np.diff(np.log(closes[-_long_n - 1:-1]))), 1e-6)
            _target_vol_dyn = 0.7 * TARGET_VOL + 0.3 * _baseline_vol
            vol_ratio = realized_vol / _target_vol_dyn

            # Vol-adaptive smoothing: more in calm (span~3), less in choppy (span~2)
            # vol_ratio < 0.7 (calm): alpha=0.5 (span=3); vol_ratio > 1.2 (choppy): alpha=0.67 (span=2)
            _smooth_alpha = 0.5 + 0.17 * max(0.0, min(1.0, (vol_ratio - 0.7) / 0.5))
            smoothed_closes = np.empty_like(closes, dtype=float)
            smoothed_closes[0] = closes[0]
            for _si in range(1, len(closes)):
                smoothed_closes[_si] = _smooth_alpha * closes[_si] + (1 - _smooth_alpha) * smoothed_closes[_si - 1]
            dyn_threshold = BASE_THRESHOLD * (0.10 + vol_ratio * 0.90) ** 0.85
            dyn_threshold = max(DYN_THRESHOLD_FLOOR, min(DYN_THRESHOLD_CEIL, dyn_threshold))

            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            dyn_threshold *= 1.0 - TREND_THRESHOLD_SCALE * (1.0 - min(abs(ret_long) / TREND_THRESHOLD_DECAY, 1.0) ** 0.85)

            _lr = linregress(np.arange(LINREG_PERIOD), np.log((bd.history["high"].values[-LINREG_PERIOD:] + bd.history["low"].values[-LINREG_PERIOD:]) / 2.0))

            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))))

            # 5-bar median for both signals (maximum noise immunity, returns sacrificed for stability)
            _med_ref_short = np.median(smoothed_closes[-SHORT_WINDOW - 2: -SHORT_WINDOW + 3])
            _med_ref_med = np.median(smoothed_closes[-adaptive_med - 2: -adaptive_med + 3])
            ret_vshort = (smoothed_closes[-1] - _med_ref_short) / _med_ref_short
            ret_short = (smoothed_closes[-1] - _med_ref_med) / _med_ref_med

            _ef, _es = ema(closes[-(EMA_SLOW+10):], EMA_FAST)[-1], ema(closes[-(EMA_SLOW+10):], EMA_SLOW)[-1]
            _ret_long_lagged = (closes[-2] - closes[-LONG_WINDOW - 1]) / closes[-LONG_WINDOW - 1]
            rsi_trend_str = min(abs(_ret_long_lagged) / RSI_TREND_BIAS_DECAY, 1.0)
            _rd = np.diff(closes[-(int(round(6 + 2 * rsi_trend_str)) + 1):])
            rsi = 100 - 100 / (1 + np.mean(np.maximum(_rd, 0)) / max(np.mean(np.maximum(-_rd, 0)), 1e-10))
            _ml = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST) - ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
            _ea = ema(closes[-(EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5):], EMA_SLOPE_PERIOD)

            # 6 voters with smooth tanh contribution: hard binary except at threshold boundary.
            # Each voter contribution = 0.5 * (1 + tanh((signal - thresh) * sharpness)) so it behaves like a binary
            # 0/1 except in a narrow band around the threshold where it transitions smoothly. Keeps original
            # vote-count semantics (sum stays in [0, 6]) while reducing flip-rate near boundaries.
            _rsi_thresh = 50 + RSI_TREND_BIAS * rsi_trend_str * (-1.0 if ret_long > 0 else 1.0)
            _macd_diff = (_ml[-1] - ema(_ml, MACD_SIGNAL)[-1]) / mid
            _ea_slope = (_ea[-1] - _ea[-EMA_SLOPE_LOOKBACK]) / _ea[-EMA_SLOPE_LOOKBACK]
            _voter_signals_bull = [
                (ret_short - dyn_threshold) / max(dyn_threshold * 0.20, 1e-6),
                (_ef - _es) / (mid * 0.0008),
                (rsi - _rsi_thresh) / 4.0,
                (_macd_diff - 0.0003) / 0.00012,
                (_lr.slope - 0.00015) / 0.00010,
                (_ea_slope - 0.0005) / 0.00025,
            ]
            # Voter contribution clipping: each conf bounded to [0.1, 0.9] instead of (0,1).
            # Prevents any single voter from dominating the strong-sum under noise saturation.
            # A noise-flipped voter shifts _bull_strong by at most ~0.8 (was ~2.0).
            _bull_confs = [0.1 + 0.8 * 0.5 * (1.0 + np.tanh(s)) for s in _voter_signals_bull]
            _bear_confs = [0.1 + 0.8 * 0.5 * (1.0 + np.tanh(-s)) for s in _voter_signals_bull]
            bull_votes = sum(_bull_confs)
            bear_votes = sum(_bear_confs)
            # Quintic-ramp strong-sum with per-voter noise-sensitivity weights.
            # Voter ordering: [ret_short, EMA_cross, RSI, MACD, slope_16, EMA_slope].
            # Weights inverse to estimated noise sensitivity (sum=6.0, preserves scale).
            _voter_weights = (0.7, 1.25, 1.10, 1.00, 0.85, 1.10)
            _bull_strong = sum(max(0.0, (c - 0.5) ** 5 * 97.66) * w for c, w in zip(_bull_confs, _voter_weights))
            _bear_strong = sum(max(0.0, (c - 0.5) ** 5 * 97.66) * w for c, w in zip(_bear_confs, _voter_weights))
            # Sideways-aware strong-sum threshold: tighten in low-trend regimes to filter
            # noisy entries; relax in trends. Uses continuous rsi_trend_str interpolation.
            _strong_min = STRONG_WEIGHT_MIN + 0.20 * (1.0 - rsi_trend_str)

            # Architectural: isolated-spike penalty on entry threshold.
            # Track last 2 bars of strong-side firings; if current strong-sum crossed
            # _strong_min but the prior 2 bars sat well below it, the crossing is
            # likely a noise spike. Penalize by tightening _strong_min proportional to
            # how far prior bars were below the firing line. Continuous: penalty =
            # 0.10 * max(0, 1 - mean_prior_above_ratio). Adds new state and history-aware
            # control flow to the entry gate.
            _bh = self._bull_strong_hist.get(symbol, [])
            _eh = self._bear_strong_hist.get(symbol, [])
            _bull_prior_ratio = sum(min(1.0, s / max(_strong_min, 1e-6)) for s in _bh) / 2.0 if len(_bh) == 2 else 1.0
            _bear_prior_ratio = sum(min(1.0, s / max(_strong_min, 1e-6)) for s in _eh) / 2.0 if len(_eh) == 2 else 1.0
            _bull_strong_min = _strong_min * (1.0 + 0.10 * max(0.0, 1.0 - _bull_prior_ratio))
            _bear_strong_min = _strong_min * (1.0 + 0.10 * max(0.0, 1.0 - _bear_prior_ratio))
            # Update history (always) — buffer of length 2.
            self._bull_strong_hist[symbol] = (_bh + [_bull_strong])[-2:]
            self._bear_strong_hist[symbol] = (_eh + [_bear_strong])[-2:]
            # Conviction margins (relative excess of strong-sum over its admission threshold).
            # Computed at top-level so they are available to both entry and flip paths.
            _bull_margin = (_bull_strong - _bull_strong_min) / max(_bull_strong_min, 1e-6)
            _bear_margin = (_bear_strong - _bear_strong_min) / max(_bear_strong_min, 1e-6)

            cooldown_trend_strength = min(abs(ret_long) / COOLDOWN_TREND_DECAY, 1.0)
            trend_avg = (TREND_GATE_MED_WEIGHT_SIDEWAYS - (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength) * ((closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]) + ((1.0 - TREND_GATE_MED_WEIGHT_SIDEWAYS) + (TREND_GATE_MED_WEIGHT_SIDEWAYS - TREND_GATE_MED_WEIGHT_BASE) * cooldown_trend_strength) * ret_long
            # Use trend_avg directly (stateless) — EMA smoothing amplifies noise via state propagation
            self.smoothed_trend[symbol] = trend_avg

            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS * cooldown_trend_strength

            calm_boost = 1.0 + CALM_BOOST_MAX * max(0.0, 1.0 - max(0.5, max(np.std(np.diff(np.log(closes[-VOL_SHORT_LOOKBACK - 1:-1]))), 1e-6) / max(np.std(np.diff(np.log(closes[-VOL_LONG_LOOKBACK - 1:-1]))), 1e-6))) ** 0.85 * min(1.0, max(0.0, (1.7 - vol_ratio) / 0.4))

            sideways_boost = 1.0 + SIDEWAYS_BOOST_MAX * (1.0 - rsi_trend_str ** 1.45)

            vol_confirm_mult = max(VOL_CONFIRM_FLOOR, min(VOL_CONFIRM_CAP, np.mean(bd.history["volume"].values[-VOL_CONFIRM_LOOKBACK:]) / np.mean(bd.history["volume"].values[-VOL_CONFIRM_BASE:])))
            strength_scale = max(0.6 + (STRENGTH_FLOOR_SIDEWAYS - 0.6) * (1.0 - min(abs(ret_long) / STRENGTH_FLOOR_DECAY, 1.0)), min(2.0, (abs(ret_short) / dyn_threshold) ** 0.85))
            combined_mult = max(0.3, min(2.5, (TARGET_VOL / realized_vol) ** 0.85)) * strength_scale * calm_boost * sideways_boost * (1.0 + CROSS_ASSET_FIXED_BOOST * (1.0 - cooldown_trend_strength)) * HIGH_VOTE_BOOST_MULT * vol_confirm_mult
            # Architectural: ADDITIONAL chop-only post-cap boost (smaller magnitude 0.08).
            # Original cross-asset boost stays inside combined_mult (preserves bull behavior
            # via cap-absorption). Additional post-cap boost activates only in deep chop
            # (trend_strength<0.3), giving sideways/rally a small additional size lift that
            # is NOT subject to cap.
            _xa_chop_gate = max(0.0, min(1.0, (0.3 - cooldown_trend_strength) / 0.2))
            _xa_boost = 1.0 + 0.08 * _xa_chop_gate
            # Architectural: smooth ONLY the upper hard ternary at vol_ratio=1.2.
            # Keep the original linear 0.6->1.2 interpolation (load-bearing for rally
            # dwell point). Replace discontinuity at vol_ratio=1.2 with smooth blend
            # to MAX_COMBINED_MULT_HIGH_VOL=2.5 over [1.2, 1.5].
            _cap_low_t = max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_LOW) / (MAX_COMBINED_VOL_HIGH - MAX_COMBINED_VOL_LOW)))
            _cap_base = MAX_COMBINED_MULT_LOW_VOL - 3.0 * _cap_low_t  # original linear interp
            # Smooth blend at upper boundary (vol_ratio > 1.2 was hard switch to 2.5)
            _cap_high_t = max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_HIGH) / 0.3))
            _cap_high_smooth = _cap_high_t * _cap_high_t * (3.0 - 2.0 * _cap_high_t)
            _cap_base = _cap_base * (1.0 - _cap_high_smooth) + MAX_COMBINED_MULT_HIGH_VOL * _cap_high_smooth
            combined_mult = min(combined_mult, _cap_base + MAX_COMBINED_TREND_BOOST * (1.0 - rsi_trend_str ** 0.85))
            size = equity * BASE_POSITION_SIZE * combined_mult * _xa_boost

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos
            _is_flip_this_bar = False

            # Architectural: vol-conditioned initial commit fraction. Continuous tanh
            # mapping vol_ratio (band ~0.5..1.5 -> ~0.50..0.36). Decouples first-bar
            # exposure from a constant in regimes where initial-bar noise risk varies.
            _entry_frac_dyn = ENTRY_INITIAL_FRAC_BASE - ENTRY_INITIAL_FRAC_VOL_AMP * np.tanh((vol_ratio - 1.0) / 0.4)
            # Architectural: directional-confluence first-bar amplifier.
            # When 16-bar log-price slope (_lr.slope) aligns in sign and magnitude
            # with the trend_avg gate, the entry is a high-confluence event — two
            # orthogonal-window signals agree. Continuous tanh on the product
            # _lr.slope * trend_avg (positive = same direction, scale by typical
            # trending magnitudes 0.0005 * 0.02 = 1e-5). One-sided positive boost
            # only (negative product stays at 0). Adds [+0.0, +0.06] to first-bar
            # frac. Does NOT couple to entry voter signals — uses two trend-window
            # primitives that are not in the strong-sum.
            # Vol-adaptive scale: in low-vol both signals shrink, so the
            # confluence threshold scales down with vol_ratio**2 to maintain
            # similar activation across regimes.
            _confluence_raw = _lr.slope * trend_avg
            _confluence_scale = 1e-5 * max(0.7, min(2.0, vol_ratio ** 2))
            _confluence_adj = 0.06 * np.tanh(max(0.0, _confluence_raw) / _confluence_scale)
            # Architectural: triple-source confluence amplifier. When EMA slope sign
            # agrees with ret_long sign (long-window EMA confirms long-window return),
            # this is a third-source agreement on top of the slope*trend_avg confluence.
            # Triple-confluence is rare and high-quality; amplify _confluence_adj by
            # smooth factor up to 1.5x. Continuous on smooth product of soft sign
            # indicators, not binary. Adds new data dependency: _confluence_adj scales
            # with cross-timescale agreement (EMA vs LR_slope vs MED2 trend_avg).
            _ea_sign_soft = np.tanh(_ea_slope / 0.0008)        # smooth sign of EMA slope
            _rl_sign_soft = np.tanh(ret_long / 0.02)           # smooth sign of long-return
            _triple_agree = max(0.0, _ea_sign_soft * _rl_sign_soft)  # both same sign
            _confluence_adj *= 1.0 + 0.5 * _triple_agree
            # Architectural: Kaufman efficiency ratio gate on initial commitment.
            # ER = |close[-1] - close[-N]| / sum(|close[i] - close[i-1]|), range [0,1].
            # High ER (>0.4) = price moved efficiently in one direction (signal-rich bars).
            # Low ER (<0.2) = path was choppy relative to net move (noise-rich, even if
            # net direction matches voters). Orthogonal to all current primitives:
            # vol_ratio measures magnitude, trend_avg measures net direction, slope measures
            # linear trajectory — ER measures path efficiency. Continuous tanh modulation
            # of _entry_frac_dyn: low ER attenuates, high ER amplifies (range -0.04..+0.04).
            # 12-bar window over smoothed_closes (already noise-attenuated for input parity).
            _er_window = 12
            _er_path = np.sum(np.abs(np.diff(smoothed_closes[-_er_window - 1:])))
            _er_net = abs(smoothed_closes[-1] - smoothed_closes[-_er_window - 1])
            _er = _er_net / max(_er_path, 1e-10)
            # One-sided deep-chop suppression: only fire on very low ER (<0.15),
            # smaller magnitude to avoid uniform size-attenuation across regimes.
            # tanh activates as ER drops below 0.15 toward 0; max attenuation -0.025.
            _er_adj = -0.025 * max(0.0, np.tanh((0.15 - _er) / 0.10))
            _entry_frac_dyn = min(0.55, _entry_frac_dyn + _confluence_adj + _er_adj)

            if current_pos == 0 and not in_cooldown:
                # Architectural simplification: removed _avg_signal bias from trend gate.
                # _avg_signal is the mean of the same 6 voter signals that drive _bull_strong/
                # _bear_strong (via _bull_confs/_bear_confs). Adding _avg_signal bias to the
                # trend gate created CORRELATED noise amplification at entry — both gates fire
                # on the same underlying noise. Using raw smoothed_trend decouples the trend
                # gate from the voter-signal subsystem; trend_avg derives from price-window
                # returns (orthogonal to per-bar voter signals).
                _trend_biased = self.smoothed_trend[symbol]
                # Architectural: replaced binary deadzone vote-tiebreak with continuous
                # strong-conviction admission. When _bull_strong significantly exceeds
                # _strong_min (margin = (strong - min) / min), the trend-sign requirement
                # softens proportionally: very strong conviction can override small-magnitude
                # wrong-sign trend. Smooth replacement for the binary deadzone clause —
                # gates on conviction magnitude rather than on absolute |_trend_biased|.
                _bull_admit = _trend_biased > -TREND_GATE_DEADZONE * min(1.0, _bull_margin / 0.3) and _trend_biased > -TREND_GATE_DEADZONE
                _bear_admit = _trend_biased < TREND_GATE_DEADZONE * min(1.0, _bear_margin / 0.3) and _trend_biased < TREND_GATE_DEADZONE
                # Architectural simplification: removed redundant bull_votes>=MIN_VOTES count gate.
                # The strong-sum gate (_bull_strong >= _bull_strong_min) is highly correlated with the
                # count gate since both derive from the same _bull_confs values. Removing the count
                # gate eliminates correlated-noise amplification at the entry decision boundary
                # (one less hard gate on the same underlying signal). Strong-sum is the primary
                # discriminator (uses voter weights and quintic ramp); count is a coarser version.
                # Architectural: conviction-margin SIZE modulation on cold entry path.
                # Symmetric to flip-path _flip_conv_adj. When the strong-sum is well above
                # its admission threshold (high conviction entry), first-bar commitment
                # is larger; marginal entries (low or negative margin) get standard size.
                # One-sided positive: only positive margin amplifies, negative is treated
                # as zero (avoids cutting size on legitimate but marginal entries near
                # the gate boundary). New data dependency: first-bar size depends on
                # conviction margin for cold entries (was independent before).
                # Concentration attenuator: count is the number of OTHER symbols already
                # on the candidate side (excludes self since current_pos==0 here).
                # Subtract own-side from counts only if own pos was non-zero (here 0).
                # Tanh on count / 1.5: count=1 -> ~0.55, count=2 -> ~0.93. Attenuation
                # magnitude up to 0.06 reduction in _entry_frac_dyn. Active only on cold entry.
                if _bull_strong >= _bull_strong_min and _bull_admit:
                    _entry_conv_adj = 0.06 * np.tanh(max(0.0, _bull_margin) / 0.30)
                    _conc_attn = 0.06 * np.tanh(_long_count_pre / 1.5)
                    target = size * min(0.55, max(0.20, _entry_frac_dyn + _entry_conv_adj - _conc_attn))
                elif _bear_strong >= _bear_strong_min and _bear_admit:
                    _entry_conv_adj = 0.06 * np.tanh(max(0.0, _bear_margin) / 0.30)
                    _conc_attn = 0.06 * np.tanh(_short_count_pre / 1.5)
                    target = -size * min(0.55, max(0.20, _entry_frac_dyn + _entry_conv_adj - _conc_attn))
            elif current_pos != 0:
                pos_pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                if current_pos < 0:
                    pos_pnl = -pos_pnl
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)

                # Position accumulation: pos_pnl-gated scale-up.
                # Architectural: scale-in ramp pace adapts to position pnl. Winning
                # scale-ins commit fully (signal was right on bar 0). Losing scale-ins
                # attenuate the per-bar increment via continuous tanh on pos_pnl scaled
                # by stop magnitude — pos_pnl=0 -> full ramp, pos_pnl=-STOP -> no further
                # scale-up (frozen at current level). New data dependency: scale-in
                # trajectory depends on realized pnl during accumulation, not just bar count.
                if bars_held <= ENTRY_FULL_BARS:
                    # Trend-agreement override: when trend_avg strongly aligns with position
                    # direction (signal still validates scale-in), bypass pnl-attenuation.
                    # Continuous tanh on (trend_avg * pos_dir) scaled by typical trending magnitude.
                    _pos_dir = 1.0 if current_pos > 0 else -1.0
                    _trend_agree = max(0.0, np.tanh(trend_avg * _pos_dir / 0.012))  # in [0,1]
                    _ramp_attn_pnl = 0.5 * (1.0 + np.tanh(pos_pnl / abs(STOP_LOSS_PCT)))  # in [0,1]
                    # Blend: full ramp when trend agrees, pnl-attenuated otherwise.
                    _ramp_attn = _trend_agree + (1.0 - _trend_agree) * _ramp_attn_pnl
                    _eff_progress = (bars_held - 1) / ENTRY_FULL_BARS + (1.0 / ENTRY_FULL_BARS) * _ramp_attn
                    _eff_progress = max(0.0, min(1.0, _eff_progress))
                    scale_frac = min(1.0, ENTRY_INITIAL_FRAC + (1.0 - ENTRY_INITIAL_FRAC) * _eff_progress)
                    full_target = size if current_pos > 0 else -size
                    target = full_target * scale_frac

                # Unified soft exit-pressure architecture (slope + peak_profit + time only).
                # Stop-loss kept as hard gate (entry-anchored, already noise-immune).
                # Architectural: confirmed-peak update — peak shifts only when pos_pnl
                # exceeds previous peak AND is rising (pos_pnl > prev_pos_pnl). Single-bar
                # noise spikes don't anchor the peak. Sideways sharpness preserved (peaks
                # confirmed within 1 extra bar). Different from EMA smoothing: this is a
                # gating rule on the high-water mark, not a low-pass filter.
                _prev_pnl = self._smoothed_pnl.get(symbol, pos_pnl)
                self._smoothed_pnl[symbol] = pos_pnl
                _curr_peak = self.peak_pnl.get(symbol, 0.0)
                # Confirmed-peak update: peak shifts only when pos_pnl > prev_peak AND
                # pos_pnl >= prev_pos_pnl (rising bar).
                if pos_pnl > _curr_peak and pos_pnl >= _prev_pnl:
                    self.peak_pnl[symbol] = pos_pnl
                else:
                    self.peak_pnl[symbol] = _curr_peak

                # Architectural: stop-loss as smooth pressure source. Vol-adaptive band width:
                # low vol (rally/sideways) -> narrow band (closer to binary, less near-stop oscillation);
                # high vol (crash) -> wide band (absorbs larger noise excursions).
                # Band half-width scales as 0.06 + 0.20*min(1, vol_ratio) of |STOP|.
                _stop_abs = abs(STOP_LOSS_PCT)
                _loss = -pos_pnl
                _band_half = (0.06 + 0.20 * min(1.0, vol_ratio)) * _stop_abs
                _sl_pressure = max(0.0, min(1.0, (_loss - (_stop_abs - _band_half)) / (2.0 * _band_half)))

                # Slope-against pressure: use MEDIAN of 3 slopes at different windows for
                # robustness. Single _lr.slope (16-bar) is shared with entry voter — coupling
                # entry & exit noise. Computing slopes at 12/16/22 and taking median decouples
                # exit-noise from entry-noise AND robust-aggregates against single-window outliers.
                # Multi-window slope MEAN (not median): mean averages out window-specific noise
                # better than median in low-vol where all 3 slopes are small and noise-dominated.
                # Median can flip on a single window; mean spreads the contribution.
                _hl2 = (bd.history["high"].values + bd.history["low"].values) / 2.0
                _slopes = []
                for _w in (12, 16, 22):
                    _ll = linregress(np.arange(_w), np.log(_hl2[-_w:]))
                    _slopes.append(_ll.slope)
                _exit_slope = float(np.mean(_slopes))
                _slope_against = -_exit_slope if current_pos > 0 else _exit_slope
                _slope_thresh = 0.0003 + 0.0003 * max(0.0, min(1.0, (0.7 - vol_ratio) / 0.3))
                _slope_band = 0.20 + 0.30 * max(0.0, min(1.0, (0.9 - vol_ratio) / 0.4))
                _sl_slope_pressure = max(0.0, min(1.0, (_slope_against - (1.0 - _slope_band/2) * _slope_thresh) / (_slope_band * _slope_thresh)))

                # Peak-profit soft pressure: vol-adaptive band (same architectural pattern as SL).
                # Low vol -> narrower band (closer to binary, less near-giveback oscillation).
                # High vol -> wider band (absorbs giveback-ratio noise from price chop).
                _pp_min = PEAK_PROFIT_MIN_BASE * max(0.6, min(2.0, vol_ratio ** 0.5))
                _giveback = max(0.0, self.peak_pnl[symbol] - pos_pnl)
                _giveback_ratio = _giveback / max(self.peak_pnl[symbol], _pp_min)
                # Architectural: profit-magnitude-aware giveback amplification.
                # When peak_pnl is large relative to _pp_min (big win), the giveback ratio
                # is amplified to lock in gains earlier (tighter trailing). When peak_pnl
                # is just above _pp_min (marginal win), no amplification. Continuous tanh
                # on (peak_pnl/_pp_min - 1.0), one-sided positive. Adds [0, 0.20] amplification
                # to _giveback_ratio. New data dependency: giveback ratio scales with
                # absolute profit magnitude not just relative giveback.
                _profit_magnitude = max(0.0, self.peak_pnl[symbol] / max(_pp_min, 1e-6) - 1.0)
                _giveback_ratio = _giveback_ratio * (1.0 + 0.18 * np.tanh(_profit_magnitude / 0.7))
                _pp_band = 0.10 + 0.20 * min(1.0, vol_ratio)
                _pp_lower = PEAK_PROFIT_GIVEBACK * (1.0 - _pp_band)
                # Architectural: smooth pp-activation ramp replacing hard binary gate.
                # Original: pp_pressure = 0 below peak == _pp_min, full ramp above. Hard
                # boundary at peak == _pp_min creates noise discontinuity in stab tests.
                # Replace with smooth tanh activation (peak_pnl/_pp_min - 1.0) scaled by 0.5,
                # giving 0.5 at peak == _pp_min and saturating to 1.0 at peak == 1.5*_pp_min.
                # This is a primitive change to pp_pressure activation: was binary gate,
                # now continuous mixture between unconditional pp_pressure and zero.
                # Trend-gated smooth activation: smooth ramp only in trending regimes
                # (rsi_trend_str high, where bull/crash benefits manifest), near-binary
                # in chop where the smoothing destabilizes peak-protection. cooldown_trend_strength
                # is bounded [0,1] and equals min(|ret_long|/0.06, 1) — well-aligned for this.
                # Narrow boundary smoothing only: linear ramp in [0.95, 1.04]*_pp_min.
                # Slightly narrower upper bound — restores baseline pp_pressure faster
                # at peak ratios above 1.04, recovering raw revenue while keeping the
                # bull-boosting smoothing in the [0.95, 1.04] band.
                _pp_ratio = self.peak_pnl[symbol] / max(_pp_min, 1e-6)
                if _pp_ratio <= 0.95:
                    _pp_activation = 0.0
                elif _pp_ratio >= 1.04:
                    _pp_activation = 1.0
                else:
                    _pp_activation = (_pp_ratio - 0.95) / 0.09
                _pp_raw = max(0.0, min(1.0, (_giveback_ratio - _pp_lower) / (PEAK_PROFIT_GIVEBACK * _pp_band)))
                _pp_pressure = _pp_raw * _pp_activation

                # Time pressure: wider smooth ramp (4 bars) to reduce noise sensitivity
                # Uses same robust median exit-slope for consistency within exit subsystem.
                _slope_agrees = (_exit_slope > 0 and current_pos > 0) or (_exit_slope < 0 and current_pos < 0)
                _slope_strength = min(1.0, abs(_exit_slope) / 0.0006)
                # Architectural: vol-conditioned symmetric momentum hold bonus.
                # Slope-against shortens max_hold but only at full strength when
                # slope is signal-dominated (high vol). In low-vol (rally chop) the
                # shortening is attenuated by min(1, vol_ratio) — slope noise in
                # rally would otherwise create noise-driven early time exits.
                # Extension (slope-agrees) remains unchanged (bull/crash extended hold).
                _short_atten = min(1.0, vol_ratio)
                _hold_adj = MOMENTUM_HOLD_BONUS * _slope_strength * (1.0 if _slope_agrees else -_short_atten)
                _max_hold = HOLD_DECAY_START + (1.0 / HOLD_DECAY_RATE) + _hold_adj
                _time_pressure = max(0.0, min(1.0, (bars_held - _max_hold + 3.0) / 4.0))

                # PnL-conditioned exit-pressure weighting (architectural change to fusion):
                # In profit (pos_pnl > 0), peak-profit dominates — preserve gains via giveback.
                # In loss (pos_pnl < 0), slope-against dominates — cut losers via momentum reversal.
                # Stop-loss and time pressure stay at unit weight (protective + structural).
                # Smooth transition via tanh of pos_pnl scaled by stop magnitude.
                _pnl_scale = np.tanh(pos_pnl / abs(STOP_LOSS_PCT))   # in [-1, 1]
                # Architectural: scale-in-aware slope-pressure attenuator. During the first
                # ENTRY_FULL_BARS bars, slope can transiently oppose position direction due
                # to micro-noise on a position not yet at full size. Attenuate _w_slope
                # smoothly with bars_held so slope-against pressure ramps up with position
                # commitment. Linear ramp from 0.5x at bar 0 to 1.0x at bar ENTRY_FULL_BARS
                # and onward. New data dependency: slope-pressure weight on bars_held.
                _scale_in_w = 0.5 + 0.5 * min(1.0, bars_held / ENTRY_FULL_BARS)
                _w_slope = (1.0 + 0.15 * max(0.0, -_pnl_scale)) * _scale_in_w  # heavier in loss, lighter during scale-in
                _w_pp    = (1.0 + 0.20 * max(0.0, _pnl_scale)) * _scale_in_w   # heavier in profit, lighter during scale-in
                # Architectural extension: time-pressure asymmetric weight by pnl_scale.
                # In profit: heavier time pressure (lock in gains via time exit).
                # In loss: lighter time pressure (give losing positions room to recover
                # before time-killing — alignment with slope-against doing the loss-cutting).
                # Asymmetric one-sided: heavier in profit (lock gains), neutral in loss
                # (let slope-against do loss-cutting; avoid sideways small-loss jitter
                # destabilizing time pressure).
                _w_time  = 1.0 + 0.20 * max(0.0, _pnl_scale)         # [-1,1] -> [1.0, 1.2]
                _exit_pressure = _sl_pressure + _w_slope * _sl_slope_pressure + _w_pp * _pp_pressure + _w_time * _time_pressure
                # Architectural: pos_pnl-gated scale-in exit threshold ramp.
                # During scale-in (bars_held <= ENTRY_FULL_BARS) AND winning (pos_pnl > 0),
                # raise the exit threshold from 1.0 to 1.2 along a smooth linear ramp
                # (1.2 at bar 0, 1.0 at bar ENTRY_FULL_BARS). Protects winning scale-in
                # from noise-driven premature exits while letting losing scale-in exit
                # normally (no protection — losing positions are noise-vulnerable too).
                # Stop-loss is exempt (full _sl_pressure forces exit regardless).
                _scale_in_winning = bars_held <= ENTRY_FULL_BARS and pos_pnl > 0
                # Architectural: 2D vol-time exit_thresh modulator combined with scale-in winning protection
                # via a single multiplicative form. _vt_factor ramps with low-vol AND mid-life.
                _vt_factor = max(0.0, min(1.0, (0.85 - vol_ratio) / 0.35)) * max(0.0, min(1.0, 1.0 - abs((bars_held - 8.0) / 6.0)))
                _exit_thresh = (1.0 + 0.20 * max(0.0, 1.0 - bars_held / ENTRY_FULL_BARS) if _scale_in_winning else 1.0) * (1.0 + 0.10 * _vt_factor)
                # Architectural: flip-origin exit-threshold protection. Positions that
                # originated from a flip are higher-conviction reversals (passed both
                # vote-count AND trend-sign AND opposite-side strong-min gates). Give
                # them extra maturation: smooth additive bonus to _exit_thresh that
                # decays linearly over the first 3 bars after flip. New state +
                # control-flow path that distinguishes flip-origin from cold-entry
                # positions in the exit-pressure decision. Stop-loss exemption below
                # already overrides this protection on real adverse moves.
                if self._from_flip.get(symbol, False):
                    _flip_age_decay = max(0.0, 1.0 - bars_held / 3.0)
                    _exit_thresh = _exit_thresh + 0.15 * _flip_age_decay
                # Stop-loss exemption: when _sl_pressure is near saturation, force standard threshold.
                if _sl_pressure >= 0.95:
                    _exit_thresh = 1.0
                if _exit_pressure >= _exit_thresh and target != 0:
                    target = 0.0

                # Flip mechanism (votes + trend_avg sign, vol-scaled)
                if not in_cooldown and ((current_pos > 0 and bear_votes >= FLIP_MIN_VOTES and _bear_strong >= _bear_strong_min and trend_avg < 0) or (current_pos < 0 and bull_votes >= FLIP_MIN_VOTES and _bull_strong >= _bull_strong_min and trend_avg > 0)):
                    _is_flip_this_bar = True
                    # Architectural: flip uses same vol-conditioned initial fraction as entry.
                    # Symmetry — flip is a first-bar commitment to a new direction (same role
                    # as entry's first bar). Anchor at _entry_frac_dyn, then scale up with
                    # vol_ratio (full flip in high-vol crash for protection; conservative
                    # flip in low-vol where noise risk dominates).
                    # Architectural: conviction-margin SIZE modulation on flip path.
                    # Larger commitment when opposite-side strong-sum is well above its
                    # admission threshold (high conviction reversal), smaller when marginal.
                    # Continuous tanh mapping of margin -> [-0.10, +0.10] additive to base
                    # flip frac. Modulates SIZE only — gates unchanged. Distinct from prior
                    # margin-as-gate attempts which filtered flips out.
                    _flip_margin = (_bear_margin if current_pos > 0 else _bull_margin)
                    # One-sided modulation: positive margin (high conviction) increases
                    # flip size, negative margin (marginal/below-threshold gate-pass) is
                    # treated as zero — avoids cutting flip size when noise drives margin
                    # negative on legitimate but marginal flips.
                    _flip_conv_adj = 0.10 * np.tanh(max(0.0, _flip_margin) / 0.30)
                    _flip_frac = min(1.0, max(0.30, _entry_frac_dyn + (1.0 - _entry_frac_dyn) * min(1.0, vol_ratio / 1.5) + _flip_conv_adj))
                    target = (-size if current_pos > 0 else size) * _flip_frac

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    for _d in (self.entry_prices, self.peak_pnl, self.entry_bar, self._smoothed_pnl):
                        _d.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                    self._from_flip.pop(symbol, None)
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol], self.peak_pnl[symbol], self.entry_bar[symbol] = mid, 0.0, self.bar_count
                    self._from_flip[symbol] = _is_flip_this_bar

        return signals
