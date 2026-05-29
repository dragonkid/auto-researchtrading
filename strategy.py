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
MACD_SIGNAL = 6  # widened from 4 to smooth MACD histogram (reduce single-bar noise impact)

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
PEAK_PROFIT_GIVEBACK = 0.25

# Sizing multipliers
BASE_POSITION_SIZE = 0.069
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

# Vote / cooldown
MIN_VOTES = 4
FLIP_MIN_VOTES = 4
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
ENTRY_INITIAL_FRAC = 0.48  # first bar: 48% of target (lower = more noise immunity at cost of returns)
ENTRY_FULL_BARS = 3  # bars to reach full position (linear scale-in over 3 bars)


class Strategy:
    def __init__(self):
        self.entry_prices, self.exit_bar, self.peak_pnl, self.entry_bar = {}, {}, {}, {}
        self.bar_count = 0
        self.smoothed_trend = {}

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
            # 2-bar EMA smoothed closes for ret_short/ret_vshort voter inputs (noise reduction)
            _smooth_alpha = 2.0 / 3.0  # span=2 EMA
            smoothed_closes = np.empty_like(closes, dtype=float)
            smoothed_closes[0] = closes[0]
            for _si in range(1, len(closes)):
                smoothed_closes[_si] = _smooth_alpha * closes[_si] + (1 - _smooth_alpha) * smoothed_closes[_si - 1]

            realized_vol = max(np.std(np.diff(np.log(closes[-VOL_LOOKBACK - 1:-1]))), 1e-6)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.10 + vol_ratio * 0.90) ** 0.85
            dyn_threshold = max(DYN_THRESHOLD_FLOOR, min(DYN_THRESHOLD_CEIL, dyn_threshold))

            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]
            dyn_threshold *= 1.0 - TREND_THRESHOLD_SCALE * (1.0 - min(abs(ret_long) / TREND_THRESHOLD_DECAY, 1.0) ** 0.85)

            _lr = linregress(np.arange(LINREG_PERIOD), np.log((bd.history["high"].values[-LINREG_PERIOD:] + bd.history["low"].values[-LINREG_PERIOD:]) / 2.0))

            adaptive_med = max(MED_WINDOW_MIN, min(MED_WINDOW_MAX, int(round(MED_WINDOW_MIN + (MED_WINDOW_MAX - MED_WINDOW_MIN) * (1.0 / max(vol_ratio, 0.5) - 0.5) / 1.5))))

            # Asymmetric median: 3-bar for fast signal, 5-bar for medium (proven sideways +0.006)
            _med_ref_short = np.median(smoothed_closes[-SHORT_WINDOW - 1: -SHORT_WINDOW + 2])
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

            bull_votes = sum([ret_short > dyn_threshold, ret_vshort > dyn_threshold * 0.75, _ef > _es, rsi > 50 + RSI_TREND_BIAS * rsi_trend_str * (-1.0 if ret_long > 0 else 1.0), (_ml[-1] - ema(_ml, MACD_SIGNAL)[-1]) / mid > 0.0003, _lr.slope > 0.00015, (_ea[-1] - _ea[-EMA_SLOPE_LOOKBACK]) / _ea[-EMA_SLOPE_LOOKBACK] > 0.0005])
            bear_votes = sum([ret_short < -dyn_threshold, ret_vshort < -dyn_threshold * 0.75, _ef < _es, rsi < 50 + RSI_TREND_BIAS * rsi_trend_str * (-1.0 if ret_long > 0 else 1.0), (_ml[-1] - ema(_ml, MACD_SIGNAL)[-1]) / mid < -0.0003, _lr.slope < -0.00015, (_ea[-1] - _ea[-EMA_SLOPE_LOOKBACK]) / _ea[-EMA_SLOPE_LOOKBACK] < -0.0005])

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
                if bull_votes >= MIN_VOTES and (self.smoothed_trend[symbol] > 0 or (abs(self.smoothed_trend[symbol]) < TREND_GATE_DEADZONE and bull_votes > bear_votes)):
                    target = size * ENTRY_INITIAL_FRAC
                elif bear_votes >= MIN_VOTES and (self.smoothed_trend[symbol] < 0 or (abs(self.smoothed_trend[symbol]) < TREND_GATE_DEADZONE and bear_votes > bull_votes)):
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

                # Stop-loss exit (noise-immune: anchored to entry_price)
                if pos_pnl < STOP_LOSS_PCT:
                    target = 0.0

                # Linreg-slope exit (simplified: no ret_long guard, pure slope reversal)
                # Removing ret_long guard eliminates a noise-sensitive boundary condition
                if target != 0 and ((current_pos > 0 and _lr.slope < -0.0003) or (current_pos < 0 and _lr.slope > 0.0003)):
                    target = 0.0

                # Peak-profit trailing exit (noise-immune: anchored to entry_price)
                if target != 0:
                    self.peak_pnl[symbol] = max(self.peak_pnl.get(symbol, 0.0), pos_pnl)
                    if self.peak_pnl[symbol] > PEAK_PROFIT_MIN_BASE * max(0.6, min(2.0, vol_ratio ** 0.5)) and self.peak_pnl[symbol] - pos_pnl > self.peak_pnl[symbol] * PEAK_PROFIT_GIVEBACK:
                        target = 0.0

                # Momentum-decay exit (soft time pressure, slope-extended)
                if target != 0 and bars_held > HOLD_DECAY_START:
                    # Slope agreement: does linreg slope support position direction?
                    _slope_agrees = (_lr.slope > 0 and current_pos > 0) or (_lr.slope < 0 and current_pos < 0)
                    _slope_strength = min(1.0, abs(_lr.slope) / 0.0006)  # normalized slope magnitude
                    # Extra hold time when slope strongly agrees
                    _effective_max = HOLD_DECAY_START + (1.0 / HOLD_DECAY_RATE) + MOMENTUM_HOLD_BONUS * _slope_strength * (1.0 if _slope_agrees else 0.0)
                    if bars_held >= _effective_max:
                        target = 0.0

                # Flip mechanism (4 votes + trend_avg sign, vol-scaled accumulation)
                if not in_cooldown and ((current_pos > 0 and bear_votes >= FLIP_MIN_VOTES and trend_avg < 0) or (current_pos < 0 and bull_votes >= FLIP_MIN_VOTES and trend_avg > 0)):
                    # In high vol (crash/trend), flip is full size for protection
                    # In low vol (sideways), flip is partial to reduce noise impact
                    _flip_frac = min(1.0, ENTRY_INITIAL_FRAC + (1.0 - ENTRY_INITIAL_FRAC) * min(1.0, vol_ratio / 1.2))
                    target = (-size if current_pos > 0 else size) * _flip_frac

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target == 0:
                    for _d in (self.entry_prices, self.peak_pnl, self.entry_bar):
                        _d.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif current_pos == 0 or (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol], self.peak_pnl[symbol], self.entry_bar[symbol] = mid, 0.0, self.bar_count

        return signals
