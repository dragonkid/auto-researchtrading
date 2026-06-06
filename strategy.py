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
MEANREV_TREND_THRESHOLD = 0.05
MEANREV_RSI_OVERSOLD = 49
MEANREV_RSI_OVERBOUGHT = 51

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
ENTRY_INITIAL_FRAC = 0.43  # first bar: 43% of target (balance noise immunity vs DD risk)
ENTRY_FULL_BARS = 3  # bars to reach full position (linear scale-in over 3 bars)


class Strategy:
    def __init__(self):
        self.entry_prices, self.exit_bar, self.peak_pnl, self.entry_bar = {}, {}, {}, {}
        self.bar_count = 0
        self.smoothed_trend = {}
        # Two prior pnl bars for confirmed-peak gate (need 2 rising bars to update).
        self._smoothed_pnl = {}
        self._prev2_pnl = {}
        # Persistence buffers: last 2 bars of strong-side firings per symbol.
        # Used to TIGHTEN _strong_min on isolated single-bar firing spikes (noise filter).
        self._bull_strong_hist = {}
        self._bear_strong_hist = {}

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, EMA_SLOPE_PERIOD + EMA_SLOPE_LOOKBACK + 5) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close
            realized_vol = max(np.std(np.diff(np.log(closes[-VOL_LOOKBACK - 1:-1]))), 1e-6)
            vol_ratio = realized_vol / TARGET_VOL

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
            # Architectural co-gate: averaged voter signal. Variance-reduced single signal that
            # acts as an additional alignment check at entry. Common-mode noise cancels in the
            # average. Adds ONE smooth boundary in parallel to existing gates rather than tightening.
            _avg_signal = sum(_voter_signals_bull) / 6.0

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
            combined_mult = min(combined_mult, (MAX_COMBINED_MULT_HIGH_VOL if vol_ratio > MAX_COMBINED_VOL_HIGH else MAX_COMBINED_MULT_LOW_VOL - 3.0 * max(0.0, min(1.0, (vol_ratio - MAX_COMBINED_VOL_LOW) / (MAX_COMBINED_VOL_HIGH - MAX_COMBINED_VOL_LOW)))) + MAX_COMBINED_TREND_BOOST * (1.0 - rsi_trend_str ** 0.85))
            size = equity * BASE_POSITION_SIZE * combined_mult

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0 and not in_cooldown:
                # _avg_signal as BIAS to trend_avg gate: instead of hard sign check on smoothed_trend,
                # require trend_avg + _avg_signal-biased to align with side. Combines two signal sources
                # (trend gate + voter signal) into one smoother boundary; common-mode noise cancels.
                _trend_biased = self.smoothed_trend[symbol] + 0.005 * np.tanh(_avg_signal)
                # Architectural: replaced binary deadzone vote-tiebreak with continuous
                # strong-conviction admission. When _bull_strong significantly exceeds
                # _strong_min (margin = (strong - min) / min), the trend-sign requirement
                # softens proportionally: very strong conviction can override small-magnitude
                # wrong-sign trend. Smooth replacement for the binary deadzone clause —
                # gates on conviction magnitude rather than on absolute |_trend_biased|.
                _bull_margin = (_bull_strong - _bull_strong_min) / max(_bull_strong_min, 1e-6)
                _bear_margin = (_bear_strong - _bear_strong_min) / max(_bear_strong_min, 1e-6)
                _bull_admit = _trend_biased > -TREND_GATE_DEADZONE * min(1.0, _bull_margin / 0.3) and _trend_biased > -TREND_GATE_DEADZONE
                _bear_admit = _trend_biased < TREND_GATE_DEADZONE * min(1.0, _bear_margin / 0.3) and _trend_biased < TREND_GATE_DEADZONE
                if bull_votes >= MIN_VOTES and _bull_strong >= _bull_strong_min and _bull_admit:
                    target = size * ENTRY_INITIAL_FRAC
                elif bear_votes >= MIN_VOTES and _bear_strong >= _bear_strong_min and _bear_admit:
                    target = -size * ENTRY_INITIAL_FRAC
                elif abs(ret_long) < MEANREV_TREND_THRESHOLD and (rsi < MEANREV_RSI_OVERSOLD or rsi > MEANREV_RSI_OVERBOUGHT):
                    target = (size if rsi < MEANREV_RSI_OVERSOLD else -size) * ENTRY_INITIAL_FRAC
            elif current_pos != 0:
                pos_pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                if current_pos < 0:
                    pos_pnl = -pos_pnl
                bars_held = self.bar_count - self.entry_bar.get(symbol, 0)

                # Position accumulation: deterministic scale-up (no vote confirmation needed)
                # Rationale: vote check during accumulation is a noise channel.
                # Entry decision was already validated on bar 0; scale-in is commitment.
                if bars_held <= ENTRY_FULL_BARS:
                    scale_frac = min(1.0, ENTRY_INITIAL_FRAC + (1.0 - ENTRY_INITIAL_FRAC) * bars_held / ENTRY_FULL_BARS)
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
                _pp_band = 0.10 + 0.20 * min(1.0, vol_ratio)
                _pp_lower = PEAK_PROFIT_GIVEBACK * (1.0 - _pp_band)
                _pp_pressure = max(0.0, min(1.0, (_giveback_ratio - _pp_lower) / (PEAK_PROFIT_GIVEBACK * _pp_band))) if self.peak_pnl[symbol] > _pp_min else 0.0

                # Time pressure: wider smooth ramp (4 bars) to reduce noise sensitivity
                # Uses same robust median exit-slope for consistency within exit subsystem.
                _slope_agrees = (_exit_slope > 0 and current_pos > 0) or (_exit_slope < 0 and current_pos < 0)
                _slope_strength = min(1.0, abs(_exit_slope) / 0.0006)
                _max_hold = HOLD_DECAY_START + (1.0 / HOLD_DECAY_RATE) + MOMENTUM_HOLD_BONUS * _slope_strength * (1.0 if _slope_agrees else 0.0)
                _time_pressure = max(0.0, min(1.0, (bars_held - _max_hold + 3.0) / 4.0))

                # PnL-conditioned exit-pressure weighting (architectural change to fusion):
                # In profit (pos_pnl > 0), peak-profit dominates — preserve gains via giveback.
                # In loss (pos_pnl < 0), slope-against dominates — cut losers via momentum reversal.
                # Stop-loss and time pressure stay at unit weight (protective + structural).
                # Smooth transition via tanh of pos_pnl scaled by stop magnitude.
                _pnl_scale = np.tanh(pos_pnl / abs(STOP_LOSS_PCT))   # in [-1, 1]
                _w_slope = 1.0 + 0.15 * max(0.0, -_pnl_scale)        # heavier in loss
                _w_pp    = 1.0 + 0.20 * max(0.0, _pnl_scale)         # heavier in profit
                _exit_pressure = _sl_pressure + _w_slope * _sl_slope_pressure + _w_pp * _pp_pressure + _time_pressure
                if _exit_pressure >= 1.0 and target != 0:
                    target = 0.0

                # Flip mechanism (votes + trend_avg sign, vol-scaled)
                if not in_cooldown and ((current_pos > 0 and bear_votes >= FLIP_MIN_VOTES and _bear_strong >= _bear_strong_min and trend_avg < 0) or (current_pos < 0 and bull_votes >= FLIP_MIN_VOTES and _bull_strong >= _bull_strong_min and trend_avg > 0)):
                    # High vol (crash): full flip for protection
                    # Moderate vol (rally/sideways): more conservative flip (noise buffer)
                    # Low vol (calm): moderate flip
                    _flip_frac = min(1.0, ENTRY_INITIAL_FRAC + (1.0 - ENTRY_INITIAL_FRAC) * min(1.0, vol_ratio / 1.5))
                    target = (-size if current_pos > 0 else size) * _flip_frac

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    for _d in (self.entry_prices, self.peak_pnl, self.entry_bar, self._smoothed_pnl, self._prev2_pnl):
                        _d.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol], self.peak_pnl[symbol], self.entry_bar[symbol] = mid, 0.0, self.bar_count

        return signals
